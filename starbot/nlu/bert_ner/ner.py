import os
import shutil
import logging
import tempfile
import threading
import tensorflow as tf
from queue import Queue
from pathlib import Path
import numpy as np

from bert import modeling
from bert.tokenization import load_vocab
from rasa.nlu.extractors import EntityExtractor
from starbot.nlu.bert_ner.model import model_fn_builder
from starbot.nlu.bert_ner.dataset import create_dataset, mark_message_with_labels, LabelMap

# for type hint
from typing import Any, List, Optional, Text, Dict
from tensorflow.contrib import tpu
from rasa.nlu.model import Metadata
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from starbot.nlu.bert_ner.dataset import Dataset, Sentence

logger = logging.getLogger(__name__)


class Config:
    # env
    use_tpu = False
    tpu_name = ""
    tpu_zone = None
    num_tpu_cores = 8  # Only used if `use_tpu` is True. Total number of TPU cores to use.
    master = None  # [Optional] TensorFlow master URL.
    gcp_project = None

    # model
    input_length = 128
    max_seq_length = 128

    # training
    save_checkpoints_steps = 1000  # How often to save the model checkpoint.
    iterations_per_loop = 1000  # How many steps to make in each estimator call.
    train_batch_size = 16
    num_train_epochs = 10
    warmup_proportion = 0.1  # Proportion of training to perform linear learning rate warmup for.
    learning_rate = 5e-5
    eval_batch_size = 8
    predict_batch_size = 1

    # io
    bert_config = "checkpoint/bert_config.json"
    init_checkpoint = "checkpoint/bert_model.ckpt"
    vocab_file = "checkpoint/vocab.txt"
    tmp_model_dir = "output/result_dir"

    # other
    dry_run = 0
    allow_interrupt = 1

    def __init__(self, config_dict):
        self.__dict__ = config_dict


class PredictServer(threading.Thread):
    estimator: tpu.TPUEstimator
    rq: Queue
    STOP = 0

    def __init__(self, estimator: tpu.TPUEstimator):
        super().__init__()
        self.rq = Queue()
        self.sessions = {}  # TODO: 添加GC机制
        self.estimator = estimator
        self.setDaemon(True)
        self._sn = 0

    def next_sn(self):
        self._sn += 1
        return str(self._sn)

    def predict(self, msg):
        sn = self.next_sn()
        msg['sn'] = sn
        response = Queue()
        self.sessions[sn] = response
        self.rq.put(msg)
        return response.get()

    def stop(self):
        self.rq.put(self.STOP)

    def _predict_input_fn_builder(self):
        def gen():
            while True:
                sample = self.rq.get()
                if sample is self.STOP:
                    break
                yield sample

        def input_fn(params):
            return tf.data.Dataset.from_generator(gen,
                                                  output_types={
                                                      'input_ids': tf.int32,
                                                      'input_mask': tf.int32,
                                                      'sn': tf.string
                                                  },
                                                  output_shapes={
                                                      'input_ids': (None, None),
                                                      'input_mask': (None, None),
                                                      'sn': (),
                                                  })
        return input_fn

    def run(self):
        input_fn = self._predict_input_fn_builder()
        for pred in self.estimator.predict(input_fn=input_fn, yield_single_examples=False):
            sn = pred['sn']
            if isinstance(sn, bytes):
                sn = sn.decode()
            q = self.sessions.pop(sn, None)
            if q:
                q.put(pred)
            else:
                logger.error('sn missing: {}'.format(sn))


class BertExtractor(EntityExtractor):
    provides = ["entities", "bert_embedding"]
    ner_labels: LabelMap
    intent_labels: LabelMap
    predictor: PredictServer
    num_ner_labels: int
    num_intent_labels: int
    vocab: Dict[str, int]
    estimator: tpu.TPUEstimator
    MODEL_DIR = "bert_ner"
    MODEL_NAME = "model.ckpt"
    CONFIG_NAME = "config.json"
    VOCAB_NAME = "vocab.txt"

    def __init__(self, component_config: Dict[Text, Any]):
        self.defaults = {k: v for k, v in vars(Config).items() if not k.startswith('__')}
        super().__init__(component_config)

    @classmethod
    def create(cls,
               component_config: Dict[Text, Any],
               config: RasaNLUModelConfig) -> Component:

        slf: BertExtractor = super(BertExtractor, cls).create(component_config, config)
        slf._prepare_for_training(config)
        return slf

    def _prepare_for_training(self, config):
        base_dir = config.get('base_dir')
        if base_dir:
            for k in ['bert_config',
                      'init_checkpoint',
                      'vocab_file',
                      'tmp_model_dir']:
                self.component_config[k] = os.path.join(base_dir, self.component_config[k])
        self.vocab = load_vocab(self.config.vocab_file)

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional[Component] = None,
             **kwargs: Any
             ) -> Component:

        if cached_component:
            return cached_component
        else:
            slf = cls(meta)
            slf._prepare_for_prediction(model_dir, meta)
            return slf

    def _prepare_for_prediction(self, model_dir, meta, load_labels=True):
        base_dir = Path(model_dir)/meta['bert_ner_dir']
        self.config.bert_config = str(base_dir/self.CONFIG_NAME)
        self.config.init_checkpoint = str(base_dir/self.MODEL_NAME)
        self.config.vocab_file = str(base_dir/self.VOCAB_NAME)
        if load_labels:
            self.ner_labels = LabelMap.load(Path(base_dir)/'ner_labels.json')
            self.intent_labels = LabelMap.load(Path(base_dir)/'intent_labels.json')
        self.vocab = load_vocab(self.config.vocab_file)
        self.estimator = self._create_estimator(meta['num_ner_labels'],
                                                meta['num_intent_labels'], 0, 0, is_training=False)
        self.predictor = PredictServer(self.estimator)
        self.predictor.start()
        # TODO: predictor线程销毁时机?

    def _create_estimator(self, num_ner_labels, num_intent_labels, num_train_steps, num_warmup_steps, is_training):
        bert_config = modeling.BertConfig.from_json_file(self.config.bert_config)
        if self.config.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (self.config.max_seq_length, bert_config.max_position_embeddings)
            )

        tpu_cluster_resolver = None
        if self.config.use_tpu and self.config.tpu_name:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                self.config.tpu_name, zone=self.config.tpu_zone, project=self.config.gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        from tensorflow.core.protobuf import rewriter_config_pb2
        session_config = tf.ConfigProto()
        off = rewriter_config_pb2.RewriterConfig.OFF
        session_config.graph_options.rewrite_options.arithmetic_optimization = off

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=self.config.master,
            model_dir=self.config.tmp_model_dir if is_training else None,
            save_checkpoints_steps=self.config.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.config.iterations_per_loop,
                num_shards=self.config.num_tpu_cores,
                per_host_input_for_training=is_per_host),
            session_config=session_config
        )

        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_ner_labels=num_ner_labels,
            num_intent_labels=num_intent_labels,
            init_checkpoint=self.config.init_checkpoint,
            learning_rate=self.config.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=self.config.use_tpu,
            use_one_hot_embeddings=self.config.use_tpu,
            max_seq_length=self.config.max_seq_length
        )

        return tf.contrib.tpu.TPUEstimator(
            use_tpu=self.config.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=self.config.train_batch_size,
            eval_batch_size=self.config.eval_batch_size,
            predict_batch_size=self.config.predict_batch_size)

    def train(self,
              training_data: TrainingData,
              cfg: RasaNLUModelConfig,
              **kwargs: Any) -> None:

        dataset = create_dataset(training_data.training_examples)
        self.ner_labels = dataset.ner_labels
        self.num_ner_labels = len(dataset.ner_labels) + 1
        self.intent_labels = dataset.intent_labels
        self.num_intent_labels = len(dataset.intent_labels)  # FIXME +1?
        all_features = self._prepare_features(dataset)
        train_examples = dataset.examples
        num_train_steps = int(
            len(train_examples) / self.config.train_batch_size * self.config.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.config.warmup_proportion)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", self.config.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = self._input_fn_builder(all_features, is_training=True, drop_remainder=True)
        self.estimator = self._create_estimator(num_ner_labels=self.num_ner_labels,
                                                num_intent_labels=self.num_intent_labels,
                                                num_train_steps=num_train_steps,
                                                num_warmup_steps=num_warmup_steps,
                                                is_training=True)
        if not self.config.dry_run:
            try:
                self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
            except KeyboardInterrupt:
                if not self.config.allow_interrupt:
                    raise

        with tempfile.TemporaryDirectory() as tempdir:
            meta = self.persist('', tempdir)
            predictor = BertExtractor.load(meta, tempdir)

            for example in training_data.training_examples:
                ir, ner = predictor._ir_and_ner(example.text)
                example.set('prediction', {
                    'ir': ir,
                    'ner': ner
                })

    def _pad(self, lst, v):
        n = self.config.input_length - len(lst)
        if n > 0:
            return lst + [v] * n
        else:
            return lst

    def _create_int_feature(self, values):
        # return tf.train.Feature(int64_list=tf.train.Int64List(value=self._pad(list(values), 0)))
        return self._pad(list(values), 0)

    def convert_tokens_to_ids(self, tokens):
        unk = self.vocab['[UNK]']
        result = []
        for token in tokens:
            result.append(self.vocab.get(token, unk))
        return result

    def _create_single_feature_from_message(self, message_text: str):
        inputs = ['[CLS]'] + list(message_text) + ['[SEP]']
        input_ids = self.convert_tokens_to_ids(self._pad(inputs, '[PAD]'))
        input_mask = self._pad([1 for _ in inputs], 0)

        features = {"input_ids": [input_ids],
                    "input_mask": [input_mask]}
        return features

    def _create_single_feature(self, example: Sentence, dataset: Dataset):
        inputs = example.chars
        input_mask = self._pad([1 for _ in inputs], 0)
        input_ids = self.convert_tokens_to_ids(self._pad(inputs, '[PAD]'))
        ner_labels = self._pad(example.labels, '[PAD]')
        ner_label_ids = dataset.ner_label2id(ner_labels)
        intent_label = example.intent
        ood_label_id = dataset.ood_label2id([intent_label])
        intent_label_id = dataset.intent_label2onehot(intent_label)
        seg_ids = [0 for _ in input_ids]

        features = {"input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": seg_ids,
                    "ner_label_ids": ner_label_ids,
                    "intent_label_id": intent_label_id,
                    "ood_label_id": ood_label_id,
                    }
        return features

    def _prepare_features(self, dataset: Dataset):
        all_features = []
        for example in dataset.examples:
            all_features.append(self._create_single_feature(example, dataset))
        return all_features

    def _input_fn_builder(self, all_features, is_training, drop_remainder):
        def input_fn(params):
            # batch_size = params["batch_size"]
            features = {
                'input_ids': tf.constant([x['input_ids'] for x in all_features]),
                'input_mask': tf.constant([x['input_mask'] for x in all_features]),
            }
            if is_training:
                features.update({
                    'segment_ids': tf.constant([x['segment_ids'] for x in all_features]),
                    'ner_label_ids': tf.constant([x['ner_label_ids'] for x in all_features]),
                    'intent_label_ids': tf.constant([x['intent_label_id'] for x in all_features]),
                    'ood_label_ids': tf.constant([x['ood_label_id'] for x in all_features]),
                })
            ds = tf.data.Dataset.from_tensor_slices(features)

            if is_training:
                ds = ds.shuffle(1000, reshuffle_each_iteration=True).repeat()
            return ds.batch(self.config.train_batch_size, drop_remainder)

        return input_fn

    def process(self, message: Message, **kwargs: Any) -> None:
        ir, ner = self._ir_and_ner(message.text)
        extracted = self.add_extractor_name(ner)
        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)
        message.set("intent", ir, add_to_output=True)

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""
        # 将bert最新checkpoint拷贝到rasa模型输出目录
        outdir = Path(model_dir) / self.MODEL_DIR
        outdir.mkdir(parents=True, exist_ok=True)
        bert_tmp = Path(self.config.tmp_model_dir)
        prefix = (bert_tmp / 'checkpoint').read_text().split(':')[-1].strip()[1:-1]

        def save(src, dst):
            logger.error('Saving {}'.format(dst))
            shutil.copy(src, dst)

        for suffix in ['.index', '.meta', '.data-00000-of-00001']:
            # output/result_dir/ch
            dst = outdir / (self.MODEL_NAME + suffix)
            save(bert_tmp / (prefix + suffix), dst)
        save(self.config.bert_config, outdir / self.CONFIG_NAME)
        save(self.config.vocab_file, outdir / self.VOCAB_NAME)

        self.ner_labels.save(outdir / 'ner_labels.json')
        self.intent_labels.save(outdir / 'intent_labels.json')

        return {
            "bert_ner_dir": self.MODEL_DIR,
            "num_ner_labels": self.num_ner_labels,
            "num_intent_labels": self.num_intent_labels,
        }

    def _ir_and_ner(self, message_text: str) -> (str, List[Dict[Text, Any]]):
        """Take a sentence and return entities in json format"""
        # <class 'dict'>:
        # {'start': 5, 'end': 7, 'value': '标间', 'entity': 'room_type',
        # 'confidence': 0.9988710946115964, 'extractor': 'ner_crf'}
        result = self.predictor.predict(self._create_single_feature_from_message(message_text))
        ner = result['ner'][0]
        #ner = ner[:len(message_text)+2]
        #crf_params = result['crf_params']
        #from tensorflow.contrib import crf
        #viterbi_seq, viterbi_score = crf.viterbi_decode(ner, crf_params)
        ir = result['ir']
        ir_label = self.intent_labels.decode(ir.tolist())[0]
        #ner_labels = self.ner_labels.decode(viterbi_seq)
        ner_labels = self.ner_labels.decode(ner)
        # print(ner)
        # print(ner_labels)
        print("message.text={}".format(message_text))
        print("is_ood={}".format(result['ir_is_ood'][0].item()))
        for l, p in zip(self.intent_labels.labels, result['ir_prob'][0]):
            bar = '#' * int(30*p)
            print("{:<32}:{:.3f} {}".format(l, p, bar))

        entities = mark_message_with_labels(message_text, ner_labels[1:])
        ir = {
            'name': ir_label,
            'confidence': 0 if result['ir_is_ood'] > 0.6 else result['ir_prob'][0][ir[0]].item()
        }
        return ir, entities

    def test_predict(self, message_text, label):
        result = self.predictor.predict(self._create_single_feature_from_message(message_text))
        labels = self.intent_labels.encode([label])
        softmax = result['softmax'][0]
        one_hot = to_one_hot(np.array(labels), one_hot_size=len(self.intent_labels))
        product = one_hot * softmax
        sum = np.sum(product)
        ir = result['intent']
        pred_label = self.intent_labels.decode(ir.tolist())[0]
        if sum < 0.9:
            print('[{:.3f} {:<30} {:<15}/{:<15}] {}'.format(sum, "#" * int(30*float(sum)), pred_label, label, message_text))

    # =========== utils ============
    @property
    def config(self):
        return Config(self.component_config)


def to_one_hot(vector, one_hot_size):
    """
    Use to convert a column vector to a 'one-hot' matrix

    Example:
        vector: [[2], [0], [1]]
        one_hot_size: 3
        returns:
            [[ 0.,  0.,  1.],
             [ 1.,  0.,  0.],
             [ 0.,  1.,  0.]]

    Parameters:
        vector (np.array): of size (n, 1) to be converted
        one_hot_size (int) optional: size of 'one-hot' row vector

    Returns:
        np.array size (vector.size, one_hot_size): converted to a 'one-hot' matrix
    """
    import numpy as np
    squeezed_vector = np.squeeze(vector, axis=-1)

    one_hot = np.zeros((squeezed_vector.size, one_hot_size))

    one_hot[np.arange(squeezed_vector.size), squeezed_vector] = 1

    return one_hot
