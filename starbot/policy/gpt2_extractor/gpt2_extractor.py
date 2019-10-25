
import json
import os
import numpy as np
import tensorflow as tf

from starbot.policy.gpt2_extractor.gpt2 import model, sample
from starbot.policy.gpt2_extractor.gpt2 import encoder
from typing import Text, Optional


def model_init(
    model_dir='models',
    model_name='355M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=20,
    temperature=1,
    top_k=1,
    top_p=0.0
):
    """
    Interactively run the model
    :model_dir=models : String, where the model is
    :model_name=355M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_dir, model_name)
    hparams = model.default_hparams()
    with open(os.path.join(model_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with graph.as_default(), sess.as_default():
        context = tf.placeholder(tf.int32, [batch_size, None])
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(model_dir, model_name))
        saver.restore(sess, ckpt)
        np.random.seed(seed)
        return graph, sess, output, enc, context


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

    def __init__(self, graph: tf.Graph, sess: tf.Session, output, enc, context):
        self.graph = graph
        self.sess = sess
        self.output = output
        self.enc = enc
        self.context = context

    @classmethod
    def load(cls,
             model_dir: Optional[Text] = None,
             ) -> 'Gpt2Extractor':
        graph, sess, output, enc, context = model_init(os.path.join(model_dir, 'models'))
        return cls(graph, sess, output, enc, context)

    def process(self, prompt: [str], message: str) -> str:
        raw_text = preprocess(prompt, message)
        batch_size = 1
        text = None
        context_tokens = self.enc.encode(raw_text)
        generated = 0
        with self.graph.as_default(), self.sess.as_default():
            out = self.sess.run(self.output, feed_dict={
                self.context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
        for i in range(batch_size):
            generated += 1
            text = self.enc.decode(out[i])

        return text
