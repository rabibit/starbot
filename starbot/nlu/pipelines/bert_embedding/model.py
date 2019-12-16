from typing import NamedTuple

import tensorflow as tf
import numpy as np
from transformers import TFBertModel
from transformers import BertConfig
from tensorflow.keras import layers


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


def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.
    Args:
      initializer_range: float, initializer range for stddev.
    Returns:
      TruncatedNormal initializer with stddev = `initializer_range`.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


class Linear(tf.keras.layers.Layer):

    def __init__(self, num_labels, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.dense1 = layers.Dense(1024)
        self.activation1 = layers.Activation(gelu)
        self.dense2 = layers.Dense(512)
        self.activation2 = layers.Activation(gelu)
        self.dense3 = layers.Dense(num_labels)
        self.activation3 = layers.Activation('softmax')

    def call(self, inputs, **kwargs):
        output = self.dense1(inputs)
        output = self.activation1(output)
        output = self.dense2(output)
        output = self.activation2(output)
        output = self.dense3(output)
        output = self.activation3(output)
        return output


class TFBertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFBertSelfAttention, self).__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = tf.keras.layers.Dense(self.all_head_size,
                                           kernel_initializer=get_initializer(config.initializer_range),
                                           name='query')
        self.key = tf.keras.layers.Dense(self.all_head_size,
                                         kernel_initializer=get_initializer(config.initializer_range),
                                         name='key')
        self.value = tf.keras.layers.Dense(self.all_head_size,
                                           kernel_initializer=get_initializer(config.initializer_range),
                                           name='value')

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        hidden_states = inputs[0]
        attention_mask = inputs[1]
        head_mask = inputs[2]

        batch_size = tf.shape(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)  # (batch size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(key_layer)[-1], tf.float32) # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFBertModel call() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer,
                                   (batch_size, -1, self.all_head_size))  # (batch_size, seq_len_q, all_head_size)

        context_layer = self.LayerNorm(context_layer)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class TFBertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFBertSelfOutput, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.hidden_size,
                                           kernel_initializer=get_initializer(config.initializer_range),
                                           name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        hidden_states = inputs[0]
        input_tensor = inputs[1]

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFBertAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFBertAttention, self).__init__(**kwargs)
        self.self_attention = TFBertSelfAttention(config, name='self')
        self.activation = layers.Activation(gelu)
        self.dense_output = TFBertSelfOutput(config, name='output')
        self.num_hidden_layers = config.num_hidden_layers

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, inputs, training=False):
        input_tensor = inputs
        attention_mask = None
        head_mask = None

        self_outputs = self.self_attention([input_tensor, attention_mask, head_mask], training=training)
        attention_output = self.dense_output([self_outputs[0], input_tensor], training=training)
        attention_output = self.activation(attention_output)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class Block(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        super(Block, self).__init__(**kwargs)

        config = BertConfig.from_pretrained('/codes/starbot/run/hugcheckpoint')
        self.attention = TFBertAttention(config)
        self.normalizer1 = layers.LayerNormalization()
        self.normalizer2 = layers.LayerNormalization()
        self.normalizer3 = layers.LayerNormalization()
        self.activation1 = layers.Activation(gelu)
        self.activation2 = layers.Activation(gelu)
        self.activation3 = layers.Activation(gelu)
        self.intermediate = layers.Dense(2 * units)
        self.dense = layers.Dense(units)

    def call(self, inputs, **kwargs):
        output1 = self.normalizer1(inputs + self.attention(inputs))
        output1 = self.activation1(output1)
        output2 = self.normalizer2(self.intermediate(output1))
        output2 = self.activation2(output2)
        output3 = self.normalizer3(output1 + self.dense(output2))
        output = self.activation3(output3)
        return output


class BertForIntentAndNer(tf.keras.Model):
    def __init__(self, num_intent_labels, num_ner_labels, **kwargs):
        super(BertForIntentAndNer, self).__init__(**kwargs)
        # self.bert = TFBertModel.from_pretrained('bert-base-chinese')
        config = BertConfig.from_pretrained('/codes/starbot/run/hugcheckpoint')
        config.output_hidden_states = True
        self.bert = TFBertModel.from_pretrained('/codes/starbot/run/hugcheckpoint', config=config)
        # self.bert.save_pretrained('/codes/starbot/run/hugcheckpoint')
        # config.save_pretrained('/codes/starbot/run/hugcheckpoint')
        # self.normalizer = layers.LayerNormalization()
        self.intent_block0 = Block(768)
        self.intent_block = Block(768)
        self.ner_block0 = Block(768)
        self.ner_block = Block(768)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.intent_linear = Linear(num_intent_labels)
        self.ner_linear = Linear(num_ner_labels)

    def call(self, inputs, **kwargs):
        bert_hiddens = self.bert(inputs)[2]
        bert_embedding = self.intent_block0(bert_hiddens[7])[0]
        bert_embedding = self.intent_block(bert_embedding)[0]
        bert_embedding = self.intent_block(bert_embedding)[0]
        bert_embedding = self.dropout1(bert_embedding, training=kwargs.get('training', False))
        intent_output = self.intent_linear(bert_embedding[:, 0])
        bert_embedding = self.ner_block0(bert_hiddens[11])[0]
        bert_embedding = self.ner_block(bert_embedding)[0]
        bert_embedding = self.ner_block(bert_embedding)[0]
        bert_embedding = self.ner_block(bert_embedding)[0]
        bert_embedding = self.dropout2(bert_embedding, training=kwargs.get('training', False))
        ner_output = self.ner_linear(bert_embedding[:, 1:])
        return [intent_output, ner_output]
        

