import os
import tensorflow as tf

from .dataset import create_dataset
from bert.tokenization import FullTokenizer
from bert import modeling
from starbot.nlu.bert_ner.model import model_fn_builder

# for type hint
from tensorflow.contrib import tpu
from rasa_nlu.training_data import Message, TrainingData
from typing import Any, List, Optional, Text, Dict
from rasa_nlu.components import Component
from rasa_nlu.model import Metadata
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.extractors import EntityExtractor


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
    bert_config = "bert_ner/checkpoint/bert_config.json"
    init_checkpoint = "bert_ner/checkpoint/bert_model.ckpt"
    vocab_file = "bert_ner/checkpoint/vocab.txt"
    tmp_model_dir = "bert_ner/output/result_dir"

    def __init__(self, config_dict):
        self.__dict__ = config_dict


class BertExtractor(EntityExtractor):
    estimator: tpu.TPUEstimator
    provides = ["entities"]
    MODEL_DIR = "bert_ner"

    def __init__(self, component_config=None):
        self.defaults = {k: v for k, v in vars(Config).items() if not k.startswith('__')}
        super().__init__(component_config)
        self.vocab = FullTokenizer(self.config.vocab_file)

    @classmethod
    def create(cls,
               component_config: Dict[Text, Any],
               config: RasaNLUModelConfig) -> Component:

        self = super(BertExtractor, cls).create(component_config, config)  # type: BertExtractor
        self.prepare_config(config)
        return self

    def prepare_config(self, config):
        base_dir = config.get('base_dir')
        if base_dir:
            for k in ['bert_config',
                      'init_checkpoint',
                      'vocab_file',
                      'tmp_model_dir']:
                self.component_config[k] = os.path.join(base_dir, self.component_config[k])

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional[Component] = None,
             **kwargs: Any
             ) -> Component:
        return super(BertExtractor, cls).load(meta,
                                              model_dir,
                                              model_metadata,
                                              cached_component,
                                              **kwargs)

    def train(self,
              training_data: TrainingData,
              cfg: RasaNLUModelConfig,
              **kwargs: Any) -> None:

        dataset = create_dataset(training_data.training_examples)
        all_features = self._prepare_features(dataset)
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

        dataset = create_dataset(training_data.training_examples)
        train_examples = dataset.examples
        num_train_steps = int(
            len(train_examples) / self.config.train_batch_size * self.config.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.config.warmup_proportion)

        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(dataset.labels) + 1,
            init_checkpoint=self.config.init_checkpoint,
            learning_rate=self.config.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=self.config.use_tpu,
            use_one_hot_embeddings=self.config.use_tpu,
            max_seq_length=self.config.max_seq_length
        )

        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=self.config.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=self.config.train_batch_size,
            eval_batch_size=self.config.eval_batch_size,
            predict_batch_size=self.config.predict_batch_size)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", self.config.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = self._input_fn_builder(all_features, is_training=True, drop_remainder=True)
        # self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    def _pad(self, lst, v):
        n = self.config.input_length - len(lst)
        if n > 0:
            return lst + [v] * n
        else:
            return lst

    def _create_int_feature(self, values):
        # return tf.train.Feature(int64_list=tf.train.Int64List(value=self._pad(list(values), 0)))
        return self._pad(list(values), 0)

    def _prepare_features(self, dataset):
        all_features = []
        for example in dataset.examples:
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
            all_features.append(features)
        return all_features

    def _input_fn_builder(self, all_features, is_training, drop_remainder):
        def input_fn(params):
            # batch_size = params["batch_size"]
            features = {
                'input_ids': tf.constant([x['input_ids'] for x in all_features]),
                'input_mask': tf.constant([x['input_mask'] for x in all_features]),
                'segment_ids': tf.constant([x['segment_ids'] for x in all_features]),
                'label_ids': tf.constant([x['label_ids'] for x in all_features]),
            }
            ds = tf.data.Dataset.from_tensor_slices(features)

            if is_training:
                ds = ds.repeat()
            return ds.batch(self.config.train_batch_size, drop_remainder)

        return input_fn

    def process(self, message: Message, **kwargs: Any) -> None:
        # <class 'dict'>:
        # {'start': 5, 'end': 7, 'value': '标间', 'entity': 'room_type',
        # 'confidence': 0.9988710946115964, 'extractor': 'ner_crf'}
        extracted = self.add_extractor_name(self._extract_entities(message))
        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""

        return {"bert_ner_dir": self.MODEL_DIR}

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Take a sentence and return entities in json format"""

        dataset = create_dataset(message.text)
        all_features = self._prepare_features(dataset)
        if self.estimator is not None:
            if self.config.use_tpu:
                # Warning: According to tpu_estimator.py Prediction on TPU is an
                # experimental feature and hence not supported here
                raise ValueError("Prediction in TPU not supported")
            predict_drop_remainder = True if self.config.use_tpu else False

            predict_input_fn = self._input_fn_builder(all_features, is_training=False,
                                                      drop_remainder=predict_drop_remainder)

            result = self.estimator.predict(input_fn=predict_input_fn)
            return result
        else:
            return []

    # =========== utils ============
    @property
    def config(self):
        return Config(self.component_config)
