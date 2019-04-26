import os
import tensorflow as tf
import logging
import json

from starbot.nlu.bert_ner.dataset import create_dataset, mark_message_with_labels
from bert.tokenization import FullTokenizer
from bert import modeling
from starbot.nlu.bert_ner.model import model_fn_builder
from pathlib import Path
import shutil

# for type hint
from tensorflow.contrib import tpu
from rasa_nlu.training_data import Message, TrainingData
from typing import Any, List, Optional, Text, Dict
from rasa_nlu.components import Component, UnsupportedLanguageError
from rasa_nlu.model import Metadata
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.extractors import EntityExtractor

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
    train_batch_size = 32
    num_train_epochs = 10
    warmup_proportion = 0.1  # Proportion of training to perform linear learning rate warmup for.
    learning_rate = 5e-5
    eval_batch_size = 8
    predict_batch_size = 8

    # io
    bert_config = "checkpoint/bert_config.json"
    init_checkpoint = "checkpoint/bert_model.ckpt"
    vocab_file = "checkpoint/vocab.txt"
    tmp_model_dir = "output/result_dir"

    def __init__(self, config_dict):
        self.__dict__ = config_dict


class BertExtractor(EntityExtractor):
    labels_map: Dict[int, str]
    labels: [str]
    num_labels: int
    vocab: FullTokenizer
    estimator: tpu.TPUEstimator
    provides = ["entities"]
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

        slf = super(BertExtractor, cls).create(component_config, config)
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
        self.vocab = FullTokenizer(self.config.vocab_file)

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

    def _prepare_for_prediction(self, model_dir, meta):
        base_dir = os.path.join(model_dir, meta['bert_ner_dir'])
        if base_dir:
            for k in ['bert_config',
                      'init_checkpoint',
                      'vocab_file',
                      ]:
                self.component_config[k] = os.path.join(base_dir, self.component_config[k])

        labels_path = Path(base_dir) / 'labels.json'
        with labels_path.open() as labels_file:
            labels = json.load(labels_file)
            self.labels_map = {i: v for i, v in enumerate(labels)}
            self.labels_map[len(labels)] = 'U'
        self.vocab = FullTokenizer(self.config.vocab_file)
        self.estimator = self.create_estimator(meta['num_labels'], 0, 0)

    def create_estimator(self, num_labels, num_train_steps, num_warmup_steps):
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

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=self.config.master,
            model_dir=self.config.tmp_model_dir,
            save_checkpoints_steps=self.config.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.config.iterations_per_loop,
                num_shards=self.config.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=num_labels,
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
        self.labels = dataset.labels
        self.num_labels = len(dataset.labels) + 1
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
        self.estimator = self.create_estimator(self.num_labels,
                                               num_train_steps=num_train_steps,
                                               num_warmup_steps=num_warmup_steps)
        self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    def _pad(self, lst, v):
        n = self.config.input_length - len(lst)
        if n > 0:
            return lst + [v] * n
        else:
            return lst

    def _create_int_feature(self, values):
        # return tf.train.Feature(int64_list=tf.train.Int64List(value=self._pad(list(values), 0)))
        return self._pad(list(values), 0)

    def _create_single_feature_from_message(self, message):
        inputs = list(message.text)
        input_ids = self.vocab.convert_tokens_to_ids(inputs)
        input_mask = [1 for _ in inputs]
        seg_ids = [0 for _ in inputs]

        features = {"input_ids": self._create_int_feature(input_ids),
                    "input_mask": self._create_int_feature(input_mask),
                    "segment_ids": self._create_int_feature(seg_ids),
                    }
        return features

    def _create_single_feature(self, example, dataset):
        inputs = [ex.char for ex in example]
        labels = [ex.label for ex in example]
        input_ids = self.vocab.convert_tokens_to_ids(inputs)
        input_mask = [1 for _ in inputs]
        label_ids = dataset.label2id(labels)
        seg_ids = [0 for _ in inputs]

        features = {"input_ids": self._create_int_feature(input_ids),
                    "input_mask": self._create_int_feature(input_mask),
                    "segment_ids": self._create_int_feature(seg_ids),
                    "label_ids": self._create_int_feature(label_ids)}
        return features

    def _prepare_features(self, dataset):
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
                    'label_ids': tf.constant([x['label_ids'] for x in all_features]),
                })
            ds = tf.data.Dataset.from_tensor_slices(features)

            if is_training:
                ds = ds.repeat()
            return ds.batch(self.config.train_batch_size, drop_remainder)

        return input_fn

    def process(self, message: Message, **kwargs: Any) -> None:
        extracted = self.add_extractor_name(self._extract_entities(message))
        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)

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
            logger.info('Saving {}'.format(dst))
            shutil.copy(src, dst)

        for suffix in ['.index', '.meta', '.data-00000-of-00001']:
            dst = outdir / (self.MODEL_NAME + suffix)
            save(bert_tmp / (prefix + suffix), outdir / dst)
        save(self.config.bert_config, outdir / self.CONFIG_NAME)
        save(self.config.vocab_file, outdir / self.VOCAB_NAME)

        labels = outdir / 'labels.json'

        with labels.open('w') as labels_file:
            labels_file.write(json.dumps(self.labels))

        return {
            "bert_ner_dir": self.MODEL_DIR,
            "num_labels": self.num_labels,
        }

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Take a sentence and return entities in json format"""
        # <class 'dict'>:
        # {'start': 5, 'end': 7, 'value': '标间', 'entity': 'room_type',
        # 'confidence': 0.9988710946115964, 'extractor': 'ner_crf'}

        if self.estimator is not None:
            if self.config.use_tpu:
                # Warning: According to tpu_estimator.py Prediction on TPU is an
                # experimental feature and hence not supported here
                raise ValueError("Prediction in TPU not supported")
            predict_drop_remainder = True if self.config.use_tpu else False
            input_fn = self._input_fn_builder([self._create_single_feature_from_message(message)],
                                              is_training=False,
                                              drop_remainder=predict_drop_remainder)
            result = list(self.estimator.predict(input_fn=input_fn))
            labels = [self.labels_map[lid] for lid in result[0]]
            return mark_message_with_labels(message.text, labels)
        else:
            return []

    # =========== utils ============
    @property
    def config(self):
        return Config(self.component_config)