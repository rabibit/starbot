from typing import NamedTuple

import tensorflow as tf
from bert import modeling
from bert import optimization


class BertNerModel:
    def __init__(self,
                 bert_config,
                 is_training,
                 input_ids,
                 input_mask,
                 segment_ids,
                 num_ner_labels,
                 ner_label_ids,
                 num_intent_labels,
                 intent_label_ids,
                 max_seq_length,
                 use_one_hot_embeddings
                 ):
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
        )

        with tf.variable_scope("ner"):
            config = NerModelConfig(
                num_layers=2,
                rnn_size=1024,
                class_size=num_ner_labels,
                sentence_length=max_seq_length
            )
            output_layer = model.get_sequence_output()
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            self.ner_model = NerModel(output_layer, ner_label_ids, num_ner_labels, config)
            self.ner_prediction = tf.argmax(self.ner_model.prediction, axis=-1)

        with tf.variable_scope("intent"):
            output_layer = model.get_pooled_output()
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            self.intent_model = IntentClassificationModel(output_layer, intent_label_ids, num_intent_labels)
            self.intent_prediction = tf.argmax(self.intent_model.prediction, axis=-1)

    @property
    def loss(self):
        return self.intent_model.loss


class NerModelConfig(NamedTuple):
    rnn_size: int
    num_layers: int
    class_size: int
    sentence_length: int


class NerModel:
    """A BiLSTM net concat to the bert model to do NER.
    """

    def __init__(self, input_layer, labels, num_labels, args):
        self.args = args

        words_used_in_sent = tf.sign(
            tf.reduce_max(
                tf.abs(input_layer),
                reduction_indices=2
            )
        )

        self.length = tf.cast(
            tf.reduce_sum(words_used_in_sent, reduction_indices=1),
            tf.int32
        )

        if tf.test.gpu_device_name():  # TODO: and not use_tpu
            LSTM = tf.keras.layers.CuDNNLSTM
        else:
            LSTM = tf.keras.layers.LSTM

        birnn = tf.keras.layers.Bidirectional(
            LSTM(self.args.rnn_size, return_sequences=True)
        )
        output = birnn(input_layer)

        weight, bias = self.weight_and_bias(2 * args.rnn_size, args.class_size)
        output = tf.reshape(output, [-1, 2 * args.rnn_size])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        self.prediction = tf.reshape(prediction, [-1, args.sentence_length, args.class_size])
        if labels is not None:
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            self.loss = self.cost(self.prediction, one_hot_labels)

    def cost(self, prediction, labels):
        cross_entropy = labels * tf.log(prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(labels), reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @staticmethod
    def weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


class IntentClassificationModel:
    """A BiLSTM net concat to the bert model to do NER.
    """
    def __init__(self, input_layer, labels, num_labels):
        hidden_size = input_layer.shape[-1].value

        weights = tf.get_variable(
            "weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        bias = tf.get_variable(
            "bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            logits = tf.matmul(input_layer, weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            if labels is not None:
                one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
                print('one_hot_labels.shape:', one_hot_labels.shape)
                print('log_probs.shape:', log_probs.shape)
                per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
                self.loss = tf.reduce_mean(per_example_loss)
            self.prediction = probabilities


def model_fn_builder(bert_config, num_ner_labels, num_intent_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, max_seq_length):

    def _model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features.get("segment_ids", None)
        ner_label_ids = features.get("ner_label_ids", [])
        intent_label_ids = features.get("intent_label_ids", 0)
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = BertNerModel(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            ner_label_ids=ner_label_ids,
            num_ner_labels=num_ner_labels,
            intent_label_ids=intent_label_ids,
            num_intent_labels=num_intent_labels,
            max_seq_length=max_seq_length,
            use_one_hot_embeddings=use_one_hot_embeddings
        )
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            tf.logging.info("**** Trainable Variables ****")

            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            logging_hook = tf.train.LoggingTensorHook({"loss": model.loss}, every_n_iter=10)
            train_op = optimization.create_optimizer(
                model.loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=model.loss,
                train_op=train_op,
                training_hooks=[logging_hook],
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    "prediction": model.prediction,
                    "softmax": model.ner_model.prediction,
                    "sn": features["sn"]
                },
                scaffold_fn=scaffold_fn
            )
        return output_spec

    return _model_fn
