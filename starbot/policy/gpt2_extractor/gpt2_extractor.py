#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from transformers import GPT2Tokenizer, PretrainedConfig,  TFGPT2LMHeadModel as TFGPT2Model
from typing import Any, Optional, Text, Dict
from pathlib import Path
import os
import time
import logging


logger = logging.getLogger(__name__)


def prepare_guide_words(items):
    def yield_item():
        while True:
            for item in items:
                yield item
    item_gen = yield_item()

    def one(fmt):
        rnd_items = items.copy()
        item = next(item_gen)
        words = fmt.format(item)
        yield f'[{",".join(rnd_items)}]|{words}={item}'
    prefix = '''[可口可乐,雪碧,百事可乐]|可口可乐啦=可口可乐
[六个核桃,可口可乐,雪碧]|核桃就好了=六个核桃
[芬达,百事可乐,七喜,可口可乐,雪碧]|就七喜吧=七喜
[王老吉,橙汁,汇源,百岁山]|百岁山=百岁山
[美年达,依云,苏打水,青岛啤酒]|青岛=青岛啤酒
[娃哈哈,北冰洋,酸梅汤]|北冰洋吧=北冰洋
[农夫果园,果粒橙,纯果乐]|我要农夫=农夫果园
[巴黎水,椰汁,豆奶,花生奶]|那就椰汁=椰汁
'''
    return prefix + '\n'.join([s for fmt in [
        "那就{}吧",
        "那就来一瓶{}吧",
        "就{}",
        "就{}啦",
        "{0}{0}",
        "{}啦",
        "{}",
        "{}",
    ] for s in one(fmt)])


def preprocess(items, utter):
    return prepare_guide_words(items) + f'\n[{",".join(items)}]|{utter}='


def get_past_shape(hparams, batch_size=None, sequence=None):
    return [hparams.n_layer, batch_size, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]


class Gpt2Extractor(object):

    MODEL_DIR = "gpt2"
    TMP_MODEL_DIR = "output/result_dir"
    LENGTH = 20
    PretrainedConfig.get_config = PretrainedConfig.to_dict

    def __init__(self, tokennizer: GPT2Tokenizer, generator):
        self.tokennizer = tokennizer
        self.generator = generator

    @classmethod
    def load(cls,
             model_dir: Optional[Text] = None,
             ) -> 'Gpt2Extractor':
        # tokennizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        # model = TFGPT2Model.from_pretrained('gpt2-medium')
        # tokennizer.save_pretrained('/codes/starbot/run/huggpt2')
        # model.save_pretrained('/codes/starbot/run/huggpt2')
        tokennizer = GPT2Tokenizer.from_pretrained(model_dir)
        start = time.time()
        # generator = tf.saved_model.load(str(Path(model_dir)/Gpt2Extractor.MODEL_DIR) + '/gpt2_saved_model')
        generator = get_model(model_dir, Gpt2Extractor.LENGTH)
        context_tokens = tf.constant(tokennizer.encode('你好小智'))[None, :]
        generator(context_tokens)
        end = time.time()
        logger.error(f'used {end - start}s to init gpt2')
        # model = TFGPT2Model.from_pretrained(model_dir)
        return cls(tokennizer, generator)

    @staticmethod
    def persist(model_dir: Text) -> None:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""
        # 将bert最新checkpoint拷贝到rasa模型输出目录
        outdir = Path(model_dir)/Gpt2Extractor.MODEL_DIR
        outdir.mkdir(parents=True, exist_ok=True)

        tokennizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = get_model(model_dir, Gpt2Extractor.LENGTH)
        tf.keras.backend.set_learning_phase(1)
        context_tokens = tf.constant(tokennizer.encode('你好小智'))[None, :]
        # model.predict(context_tokens)
        # os.system(f'rm -rf {Gpt2Extractor.TMP_MODEL_DIR}/*')
        # model.save(Gpt2Extractor.TMP_MODEL_DIR + '/gpt2_saved_model', save_format='tf')
        # os.system(f'mv {Gpt2Extractor.TMP_MODEL_DIR}/gpt2_saved_model {outdir}')

    def process(self, prompt: [str], message: str) -> str:
        raw_text = preprocess(prompt, message)
        context_tokens = tf.constant(self.tokennizer.encode(raw_text))[None, :]
        start = time.time()
        ids = self.generator(context_tokens)
        end = time.time()
        logger.error(f'used {end - start}s to get ner')
        text = self.tokennizer.decode(ids.numpy().tolist()[0])[len(raw_text):]

        return text


def get_model(model_dir, length):
    model = TFGPT2Model.from_pretrained(model_dir)

    class generate_sequence(tf.keras.Model):
        def __init__(self, model, length):
            super(generate_sequence, self).__init__()
            self.model = model
            self.length = length

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                                        tf.TensorSpec(shape=get_past_shape(model.config), dtype=tf.float32)])
        def _predict(self, inputs, past):
            generated = inputs
            past = tf.unstack(past)
            past = tuple(past)
            for _ in tf.range(self.length - 1):
                outputs, past = self.model(generated[:, -1:], past=past)
                next_token_logits = outputs[:, -1, :]
                next_token = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)
                next_token = tf.reshape(next_token, (1, 1))
                generated = tf.concat((generated, next_token), axis=-1)
            return generated

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
        def call(self, inputs):
            generated = inputs
            outputs, past = self.model(generated)
            next_token_logits = outputs[:, -1, :]
            next_token = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)
            next_token = tf.reshape(next_token, (1, 1))
            generated = tf.concat((generated, next_token), axis=-1)
            return self._predict(generated, tf.stack(past))

    return generate_sequence(model, length)
