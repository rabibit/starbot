from typing import NamedTuple

import tensorflow as tf
from transformers import TFBertModel


class BertForIntentAndNer(tf.keras.Model):
    def __init__(self, num_intent_labels, num_ner_labels):
        super(BertForIntentAndNer, self).__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-chinese')
        self.intent = tf.keras.layers.Dense(num_intent_labels)
        self.intent_activation = tf.keras.layers.Activation('softmax', name='intent')
        self.ner = tf.keras.layers.Dense(num_ner_labels)
        self.ner_activation = tf.keras.layers.Activation('softmax', name='ner')

    def call(self, inputs):
        bert_embedding = self.bert(inputs)
        intent_dense = self.intent(bert_embedding)
        intent_output = self.intent_activation(intent_dense)
        ner_dense = self.ner(bert_embedding)
        ner_output = self.ner_activation(ner_dense)
        return [intent_output, ner_output]
        

