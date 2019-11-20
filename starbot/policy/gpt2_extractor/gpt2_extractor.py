#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel as TFGPT2Model
from typing import Any, Optional, Text, Dict
from pathlib import Path
import os


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


class Gpt2Extractor(object):

    MODEL_DIR = "gpt2"
    TMP_MODEL_DIR = "output/result_dir"

    def __init__(self, tokennizer: GPT2Tokenizer, generator, length: int):
        self.tokennizer = tokennizer
        self.generator = generator
        self.length = length

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             ) -> 'Gpt2Extractor':
        # tokennizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        # model = TFGPT2Model.from_pretrained('gpt2-medium')
        # tokennizer.save_pretrained('/codes/starbot/run/huggpt2')
        # model.save_pretrained('/codes/starbot/run/huggpt2')
        tokennizer = GPT2Tokenizer.from_pretrained('/codes/starbot/run/huggpt2')
        generator = tf.saved_model.load(str(Path(model_dir)/meta['gpt2']) + '/gpt2_saved_model')
        # model = TFGPT2Model.from_pretrained(model_dir)
        return cls(tokennizer, generator, 20)

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""
        # 将bert最新checkpoint拷贝到rasa模型输出目录
        outdir = Path(model_dir)/self.MODEL_DIR
        outdir.mkdir(parents=True, exist_ok=True)

        model = generate_sequence('/codes/starbot/run/huggpt2')
        os.system(f'rm -rf {self.TMP_MODEL_DIR}/*')
        model.save(self.TMP_MODEL_DIR + '/gpt2_saved_model', save_format='tf')
        os.system(f'mv {self.TMP_MODEL_DIR}/gpt2_saved_model {outdir}')

        return {
            "gpt2": self.MODEL_DIR,
        }

    def process(self, prompt: [str], message: str) -> str:
        raw_text = preprocess(prompt, message)
        context_tokens = tf.constant(self.tokennizer.encode(raw_text))[None, :]
        ids = self.generator(context_tokens)
        text = self.tokennizer.decode(ids.numpy().tolist()[0])[len(raw_text):]

        return text


class generate_sequence(tf.keras.Model):
    def __init__(self, model_dir):
        super(generate_sequence, self).__init__()
        self.model = TFGPT2Model.from_pretrained(model_dir)

    @tf.function
    def call(self, inputs):
        generated = inputs
        for _ in tf.range(self.length):
            outputs = self.model(generated)
            next_token_logits = outputs[0][:, -1, :]
            next_token = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)
            next_token = tf.reshape(next_token, (1, 1))
            generated = tf.concat((generated, next_token), axis=-1)
        return generated
