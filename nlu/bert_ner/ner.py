from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Any, List

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from .dataset import create_dataset
from .bert.tokenization import FullTokenizer
from .bert import modeling
import tensorflow as tf


class BertExtractor(EntityExtractor):
    name = "ner_bert"
    provides = ["entities"]
    input_length = 128
    max_seq_length = 128

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        self.vocab = FullTokenizer('./bert_ner/checkpoint/vocab.txt')  # todo
        self.dataset = create_dataset(training_data.training_examples)
        filename = "train.tf_record"  # todo
        self._prepare_features(filename)

    def _pad(self, lst, v):
        n = self.input_length - len(lst)
        if n > 0:
            return lst + [v] * n
        else:
            return lst

    def _create_int_feature(self, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=self._pad(list(values), 0)))

    def _prepare_features(self, filename):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for example in self.dataset.examples:
                inputs = [ex.char for ex in example]
                labels = [ex.label for ex in example]
                input_ids = self.vocab.convert_tokens_to_ids(inputs)
                input_mask = [1 for _ in inputs]
                label_ids = self.dataset.label2id(labels)
                seg_ids = [0 for _ in inputs]

                features = {"input_ids": self._create_int_feature(input_ids),
                            "input_mask": self._create_int_feature(input_mask),
                            "segment_ids": self._create_int_feature(seg_ids),
                            "label_ids": self._create_int_feature(label_ids)}
                record = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(record.SerializeToString())

    def _create_model(self, bert_config, is_training, input_ids, input_mask,
                      segment_ids, labels, num_labels, use_one_hot_embeddings):
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )

        output_layer = model.get_sequence_output()

        hidden_size = output_layer.shape[-1].value

        output_weight = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer()
        )
        with tf.variable_scope("loss"):
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            output_layer = tf.reshape(output_layer, [-1, hidden_size])
            logits = tf.matmul(output_layer, output_weight, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [-1, self.max_seq_length, 11])
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_sum(per_example_loss)
            probabilities = tf.nn.softmax(logits, axis=-1)
            predict = tf.argmax(probabilities, axis=-1)
            return loss, per_example_loss, logits, predict

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        pass

    def persist(self, model_dir):
        pass

    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        pass
