from typing import NamedTuple

import tensorflow as tf
import numpy as np
from transformers import TFBertModel


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    cdf: 累积分布函数(Cumulative Distribution Function)，又叫分布函数
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf
class BertForIntentAndNer(tf.keras.Model):
    def __init__(self, num_intent_labels, num_ner_labels):
        super(BertForIntentAndNer, self).__init__()
        # self.bert = TFBertModel.from_pretrained('bert-base-chinese')
        self.bert = TFBertModel.from_pretrained('/codes/starbot/run/hugcheckpoint')
        # self.bert.save_pretrained('/codes/starbot/run/hugcheckpoint')
        self.intent1 = tf.keras.layers.Dense(1024)
        self.intent_activation1 = tf.keras.layers.Activation(gelu)
        self.intent2 = tf.keras.layers.Dense(512)
        self.intent_activation2 = tf.keras.layers.Activation(gelu)
        self.intent3 = tf.keras.layers.Dense(num_intent_labels)
        self.intent_activation3 = tf.keras.layers.Activation('softmax', name='intent')
        self.ner1 = tf.keras.layers.Dense(1024)
        self.ner_activation1 = tf.keras.layers.Activation(gelu)
        self.ner2 = tf.keras.layers.Dense(512)
        self.ner_activation2 = tf.keras.layers.Activation(gelu)
        self.ner3 = tf.keras.layers.Dense(num_ner_labels)
        self.ner_activation3 = tf.keras.layers.Activation('softmax', name='ner')

    @tf.function
    def call(self, inputs):
        bert_embedding = self.bert(inputs)[0]
        intent_dense = self.intent1(bert_embedding[:, 0])
        intent_dense = self.intent_activation1(intent_dense)
        intent_dense = self.intent2(intent_dense)
        intent_dense = self.intent_activation2(intent_dense)
        intent_dense = self.intent3(intent_dense)
        intent_output = self.intent_activation3(intent_dense)
        ner_dense = self.ner1(bert_embedding[:, 1:])
        ner_dense = self.ner_activation1(ner_dense)
        ner_dense = self.ner2(ner_dense)
        ner_dense = self.ner_activation2(ner_dense)
        ner_dense = self.ner3(ner_dense)
        ner_output = self.ner_activation3(ner_dense)
        return [intent_output, ner_output]
        

