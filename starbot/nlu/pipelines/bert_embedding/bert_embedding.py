import os
import shutil
import logging
import tempfile
import threading
import collections
import tensorflow as tf
from queue import Queue
from pathlib import Path
import numpy as np
import sklearn.metrics as sk
import datetime

from rasa.nlu.extractors import EntityExtractor
from starbot.nlu.pipelines.bert_embedding.model import BertForIntentAndNer
from starbot.nlu.pipelines.bert_embedding.dataset import create_dataset, mark_message_with_labels, LabelMap

# for type hint
from typing import Any, List, Optional, Text, Dict
from rasa.nlu.model import Metadata
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from starbot.nlu.pipelines.bert_embedding.dataset import Dataset, Sentence

logger = logging.getLogger(__name__)


def split_samples(smaples):
    sample_indices = {328, 869, 27, 833, 416, 498, 239, 122, 399, 769, 307, 285, 859, 191, 329, 764, 36, 265, 90, 851,
                      457, 87, 204, 434, 10, 862, 895, 702, 585, 217, 62, 289, 75, 693, 747, 798, 246, 305, 740, 14,
                      720, 810, 882, 220, 497, 711, 814, 18, 523, 801, 886, 242, 406, 163, 201, 452, 135, 712, 95, 758,
                      319, 448, 856, 576, 468, 118, 342, 24, 797, 700, 642, 268, 326, 335, 582, 164, 803, 349, 107, 262,
                      841, 549, 578, 887, 385, 331, 577, 340, 922, 762, 485, 104}

    train = []
    evaluation = []
    for i, sample in enumerate(smaples):
        if i in sample_indices or i < 2:
            evaluation.append(sample)
        else:
            train.append(sample)

    return train, evaluation


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab


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
    do_embedding = 0

    def __init__(self, config_dict):
        self.__dict__ = config_dict


def f1(t, p):
    score = sk.f1_score(t.numpy(), p.numpy(), average='macro')
    return score


def f1_score(y, pred):
    y = tf.argmax(y, axis=-1)
    y = tf.reshape(y, [-1])
    pred = tf.argmax(pred, axis=-1)
    pred = tf.reshape(pred, [-1])
    score = tf.py_function(f1, [y, pred], tf.float32)
    return score


class RocAndPRCallback(tf.keras.callbacks.Callback):
    """get the UNROC and UNPR

    Arguments:
        eval_x: the evaluation inputs
        eval_y: the evaluation labels
        ood_id: use to detect the ood
    """

    def __init__(self, eval_x, eval_y, ood_id):
        super(RocAndPRCallback, self).__init__()
        self.eval_x = eval_x
        self.eval_y = eval_y
        self.ood_id = tf.constant(ood_id, dtype=tf.int64)

    def on_epoch_end(self, epoch, logs=None):
        logits = self.model(self.eval_x)[0]
        logits_right = tf.boolean_mask(logits, tf.equal(tf.argmax(logits, -1), tf.argmax(self.eval_y[0], -1)))
        s_right = logits_right
        s_right_prob = tf.reduce_max(s_right, axis=-1, keepdims=True)
        logits_wrong = tf.boolean_mask(logits, tf.not_equal(tf.argmax(logits, -1), tf.argmax(self.eval_y[0], -1)))
        s_wrong = logits_wrong
        s_wrong_prob = tf.reduce_max(s_wrong, axis=-1, keepdims=True)
        labels = np.zeros((s_right_prob.shape[0] + s_wrong_prob.shape[0]), dtype=np.int32)
        labels[:s_right_prob.shape[0]] += 1
        examples = np.squeeze(np.vstack((tf.reshape(s_right_prob, (-1, 1)), tf.reshape(s_wrong_prob, (-1, 1)))))
        logger.error(f'examples: {examples}')
        logger.error(f'epoch {epoch}, AUROC for misclassified: {sk.roc_auc_score(labels, examples)}')
        logger.error(f'epoch {epoch}, AUPR for misclassified: {sk.average_precision_score(labels, examples)}')
        logger.error(f'detect the ood')
        mul_factor = tf.cast(tf.not_equal(self.ood_id, tf.argmax(logits, -1)), dtype=tf.float32)
        mul_factor = tf.reshape(mul_factor, [-1, 1])
        logits = logits * mul_factor
        logits_right = tf.boolean_mask(logits, tf.equal(self.ood_id, tf.argmax(self.eval_y[0], -1)))
        s_right = logits_right
        s_right_prob = tf.reduce_max(s_right, axis=-1, keepdims=True)
        logits_wrong = tf.boolean_mask(logits, tf.not_equal(self.ood_id, tf.argmax(self.eval_y[0], -1)))
        s_wrong = logits_wrong
        s_wrong_prob = tf.reduce_max(s_wrong, axis=-1, keepdims=True)
        labels = np.zeros((s_right_prob.shape[0] + s_wrong_prob.shape[0]), dtype=np.int32)
        labels[:s_right_prob.shape[0]] += 1
        logger.error(f'labels: {labels}')
        examples = np.squeeze(np.vstack((tf.reshape(s_right_prob, (-1, 1)), tf.reshape(s_wrong_prob, (-1, 1)))))
        examples = -1 * examples
        logger.error(f'examples: {examples}')
        logger.error(f'epoch {epoch}, AUROC for ood: {sk.roc_auc_score(labels, examples)}')
        logger.error(f'epoch {epoch}, AUPR for ood: {sk.average_precision_score(labels, examples)}')


class BertEmbedding(EntityExtractor):
    provides = ["entities", "bert_embedding"]
    ner_labels: LabelMap
    modify_info_labels: LabelMap
    intent_labels: LabelMap
    predictor: BertForIntentAndNer
    num_ner_labels: int
    num_modify_info_labels: int
    num_intent_labels: int
    vocab: Dict[str, int]
    MODEL_DIR = "bert_ner"
    MODEL_NAME = "model.ckpt"
    CONFIG_NAME = "config.json"
    VOCAB_NAME = "vocab.txt"
    tmp_model_dir: tempfile.TemporaryDirectory = None

    def __init__(self, component_config: Dict[Text, Any]):
        self.defaults = {k: v for k, v in vars(Config).items() if not k.startswith('__')}
        super().__init__(component_config)

    @classmethod
    def create(cls,
               component_config: Dict[Text, Any],
               config: RasaNLUModelConfig) -> Component:

        slf: BertEmbedding = super(BertEmbedding, cls).create(component_config, config)
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
             ) -> 'BertEmbedding':

        if cached_component:
            return cached_component
        else:
            slf = cls(meta)
            slf._prepare_for_prediction(model_dir, meta)
            return slf

    def __del__(self):
        if self.tmp_model_dir:
            self.tmp_model_dir.cleanup()

    def _prepare_for_prediction(self, model_dir, meta, load_labels=True):
        # The model_dir will be removed by rasa after loaded.
        # So we need to move it to keep it live
        self.tmp_model_dir = tempfile.TemporaryDirectory()
        base_dir = Path(self.tmp_model_dir.name) / meta['bert_ner_dir']
        logger.info(f'moving bert model to {base_dir}')
        shutil.move(str(Path(model_dir) / meta['bert_ner_dir']), base_dir)

        self.config.bert_config = str(base_dir / self.CONFIG_NAME)
        self.config.init_checkpoint = str(base_dir / self.MODEL_NAME)
        self.config.vocab_file = str(base_dir / self.VOCAB_NAME)
        if load_labels:
            self.ner_labels = LabelMap.load(Path(base_dir) / 'ner_labels.json')
            self.modify_info_labels = LabelMap.load(Path(base_dir) / 'modify_info_labels.json')
            self.intent_labels = LabelMap.load(Path(base_dir) / 'intent_labels.json')
        self.vocab = load_vocab(self.config.vocab_file)
        self.predictor = tf.saved_model.load(str(base_dir / 'saved_model'))

    def train(self,
              training_data: TrainingData,
              cfg: RasaNLUModelConfig,
              **kwargs: Any) -> None:

        dataset = create_dataset(training_data.training_examples)
        self.ner_labels = dataset.ner_labels
        self.num_ner_labels = len(dataset.ner_labels)
        self.modify_info_labels = dataset.modify_info_labels
        self.num_modify_info_labels = len(dataset.modify_info_labels)
        self.intent_labels = dataset.intent_labels
        self.num_intent_labels = len(dataset.intent_labels)  # FIXME +1?
        all_features = self._prepare_features(dataset)
        train_examples = dataset.examples
        num_train_steps = int(
            len(train_examples) / self.config.train_batch_size * self.config.num_train_epochs)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", self.config.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        intent_ner_model = BertForIntentAndNer(self.num_intent_labels, self.num_ner_labels, self.num_modify_info_labels)
        if os.path.exists(self.config.tmp_model_dir):
            intent_ner_model.load_weights(self.config.tmp_model_dir + '/saved_model_weights')
        intent_ner_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-08, clipnorm=1.0),
                                 loss={'output_1': 'categorical_crossentropy',
                                       'output_2': 'categorical_crossentropy',
                                       'output_3': 'categorical_crossentropy'},
                                 loss_weights={'output_1': 1.0, 'output_2': 50.0, 'output_3': 50.0},
                                 metrics=['accuracy', f1_score])
        training_data, eval_data = split_samples(all_features)
        train_x, train_y = self._batch_generator(training_data)
        eval_x, eval_y = self._batch_generator(eval_data)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_output_2_f1_score', mode='max',
                                                          min_delta=0.05, verbose=1, patience=50,
                                                          restore_best_weights=True)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensor_board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)

        ood_id = dataset.intent_label2id(['other'])[0]
        intent_ner_model.fit(train_x, train_y, validation_data=[eval_x, eval_y], epochs=self.config.num_train_epochs,
                             callbacks=[early_stopping, tensor_board, RocAndPRCallback(eval_x, eval_y, ood_id)])
        self.model = intent_ner_model

    def _pad(self, lst, v):
        n = self.config.input_length - len(lst)
        if n > 0:
            return lst + [v] * n
        else:
            return lst

    def _fenci_pad(self, fenci_vec):
        n = self.config.input_length - len(fenci_vec)
        dim = len(fenci_vec[0])
        if n > 0:
            return fenci_vec + [[0] * dim] * n
        else:
            return fenci_vec

    def _create_int_feature(self, values):
        # return tf.train.Feature(int64_list=tf.train.Int64List(value=self._pad(list(values), 0)))
        return self._pad(list(values), 0)

    def convert_tokens_to_ids(self, tokens):
        unk = self.vocab['[UNK]']
        result = []
        for token in tokens:
            result.append(self.vocab.get(token, unk))
        return result

    def _create_single_feature_from_message(self, message_text: str, fenci_vec: List[List[int]]):
        inputs = ['[CLS]'] + list(message_text) + ['[SEP]']
        input_ids = self.convert_tokens_to_ids(self._pad(inputs, '[PAD]'))
        input_mask = self._pad([1 for _ in inputs], 0)
        fenci_vec = self._fenci_pad(fenci_vec)

        # features = {"input_ids": [input_ids],
        #             "input_mask": [input_mask]}
        # return features
        return [tf.constant([input_ids]), tf.constant([fenci_vec], dtype=tf.float32)]

    def _create_single_feature(self, example: Sentence, dataset: Dataset):
        inputs = example.chars
        input_mask = self._pad([1 for _ in inputs], 0)
        input_ids = self.convert_tokens_to_ids(self._pad(inputs, '[PAD]'))
        ner_labels = self._pad(example.labels, '[PAD]')
        ner_label_ids = dataset.ner_label2onehot(ner_labels)
        modify_info_labels = self._pad(example.modify_info_labels, '[PAD]')
        modify_info_label_ids = dataset.modify_info_label2onehot(modify_info_labels)
        intent_label = example.intent
        ood_label_id = dataset.ood_label2id([intent_label])
        intent_label_id = dataset.intent_label2onehot(intent_label)
        seg_ids = [0 for _ in input_ids]
        fenci_vec = self._fenci_pad(example.fenci_vec)

        features = {"input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": seg_ids,
                    "ner_label_ids": ner_label_ids,
                    "intent_label_id": intent_label_id,
                    "ood_label_id": ood_label_id,
                    "fenci_vec": fenci_vec,
                    "modify_info_label_ids": modify_info_label_ids,
                    }
        return features

    def _prepare_features(self, dataset: Dataset):
        all_features = []
        for example in dataset.examples:
            all_features.append(self._create_single_feature(example, dataset))
        return all_features

    def _batch_generator(self, all_features):
        x = []
        input_fencis = []
        y1 = []
        y2 = []
        y3 = []
        np.random.shuffle(all_features)
        for feature in all_features:
            x.append(feature["input_ids"])
            input_fencis.append(feature["fenci_vec"])
            y1.append(feature["intent_label_id"])
            y2.append(feature["ner_label_ids"])
            y3.append(feature["modify_info_label_ids"])
        # return tf.constant(x, dtype=tf.int32), {'intent': tf.constant(y1, dtype=tf.int32),
        return [tf.constant(x), tf.constant(input_fencis, dtype=tf.float32)], [tf.constant(y1), tf.constant(y2), tf.constant(y3)]

    def process(self, message: Message, **kwargs: Any) -> None:
        ir, ner, modify_infos, embedding = self._predict(message.text, message.get("tokens"))
        extracted = self.add_extractor_name(ner)
        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)
        extracted = self.add_extractor_name(modify_infos)
        message.set("modify_info", message.get("modify_info", []) + extracted,
                    add_to_output=True)
        message.set("intent", ir, add_to_output=True)
        message.set("bert_embedding", embedding, add_to_output=True)

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""
        # ???bert??????checkpoint?????????rasa??????????????????
        outdir = Path(model_dir) / self.MODEL_DIR
        outdir.mkdir(parents=True, exist_ok=True)

        # bert_tmp = Path(self.config.tmp_model_dir)

        def save(src, dst):
            logger.error('Saving {}'.format(dst))
            shutil.copy(src, dst)

        os.system(f'rm -rf {self.config.tmp_model_dir}/*')
        self.model.save(self.config.tmp_model_dir + '/saved_model', save_format='tf')
        os.system(f'mv {self.config.tmp_model_dir}/saved_model {outdir}')
        self.model.save_weights(self.config.tmp_model_dir + '/saved_model_weights', save_format='tf')
        save(self.config.bert_config, outdir / self.CONFIG_NAME)
        save(self.config.vocab_file, outdir / self.VOCAB_NAME)

        self.ner_labels.save(outdir / 'ner_labels.json')
        self.modify_info_labels.save(outdir / 'modify_info_labels.json')
        self.intent_labels.save(outdir / 'intent_labels.json')

        return {
            "bert_ner_dir": self.MODEL_DIR,
            "num_ner_labels": self.num_ner_labels,
            "num_modify_info_labels": self.num_modify_info_labels,
            "num_intent_labels": self.num_intent_labels,
        }

    def _predict(self, message_text: str, fenci_vec: List[List[int]]) -> (str, List[Dict[Text, Any]]):
        """Take a sentence and return entities in json format"""
        result = self.predictor(self._create_single_feature_from_message(message_text, fenci_vec), training=False)
        ner = result[1]
        ner_indexs = np.argmax(ner, axis=-1)
        modify_info = result[2]
        modify_info_indexs = np.argmax(modify_info, axis=-1)
        ir = result[0]
        ir_index = np.argmax(ir, axis=-1)
        ir_confidence = np.max(ir, axis=-1)
        print(f"ir indexs: {ir_index.tolist()}")
        print(f"ner indexs: {ner_indexs.tolist()}")
        ir_label = self.intent_labels.decode(ir_index.tolist())[0]
        ner_labels = self.ner_labels.decode(ner_indexs.tolist()[0])
        modify_info_labels = self.modify_info_labels.decode(modify_info_indexs.tolist()[0])
        print("message.text={}".format(message_text))
        for l, p in zip(self.intent_labels.labels, ir[0]):
            bar = '#' * int(30 * p)
            print("{:<32}:{:.3f} {}".format(l, p, bar))

        entities = mark_message_with_labels(message_text, ner_labels)
        entities = merge_entities(entities)
        modify_infos = mark_message_with_labels(message_text, modify_info_labels)
        modify_infos = merge_entities(modify_infos)
        ir = {
            'name': ir_label,
            'confidence': ir_confidence.item()
        }
        return ir, entities, modify_infos, [0.0] * 768

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
            print('[{:.3f} {:<30} {:<15}/{:<15}] {}'.format(sum, "#" * int(30 * float(sum)), pred_label, label,
                                                            message_text))

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


def merge_entities(entities: []) -> list:
    """
    
    :param entities: 
    :return: 
    
    >>> merge_entities([])
    []
    >>> merge_entities([{'start': 0, 'end': 2, 'entity': 'goods', 'value': '??????'}])
    [{'start': 0, 'end': 2, 'entity': 'goods', 'value': '??????'}]
    >>> merge_entities([{'start': 0, 'end': 2, 'entity': 'goods', 'value': '??????'},
    ... {'start': 2, 'end': 4, 'entity': 'goods', 'value': '??????'}])
    [{'start': 0, 'end': 4, 'entity': 'goods', 'value': '????????????'}]
    >>> merge_entities([{'start': 0, 'end': 2, 'entity': 'goods', 'value': '??????'},
    ... {'start': 3, 'end': 5, 'entity': 'goods', 'value': '??????'}])
    [{'start': 0, 'end': 2, 'entity': 'goods', 'value': '??????'}, {'start': 3, 'end': 5, 'entity': 'goods', 'value': '??????'}]
    >>> merge_entities([{'start': 0, 'end': 2, 'entity': 'goods', 'value': '??????'},
    ... {'start': 2, 'end': 4, 'entity': 'goods', 'value': '??????'},
    ... {'start': 4, 'end': 6, 'entity': 'goods', 'value': '??????'}])
    [{'start': 0, 'end': 4, 'entity': 'goods', 'value': '??????????????????'}]
    """
    if not entities:
        return []
    result = []
    entities_merged = entities[0]
    for entity in entities[1:]:
        if entities_merged['entity'] == entity['entity'] and entities_merged['end'] == entity['start']:
            entities_merged = {
                "start": entities_merged['start'],
                "end": entity['end'],
                "entity": entities_merged['entity'],
                "value": entities_merged['value'] + entity['value'],
            }
        else:
            result.append(entities_merged)
            entities_merged = entity
    result.append(entities_merged)
    return result
