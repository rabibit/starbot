#!/usr/bin/env python

from starbot.nlu import preparemd

from rasa_nlu.train import train as rasa_train
from rasa_nlu import config, utils
import logging
import os


def train():
    utils.configure_colored_logging(logging.INFO)
    preparemd.convert("nlu.md", "tmp-nlu.md", multi=1)
    nlu_config = config.load('nlu_config.yml')
    nlu_config.override({
        "base_dir": os.path.dirname(os.path.abspath(__file__))
    })

    rasa_train(
        nlu_config=nlu_config,
        data='tmp-nlu.md',
        path='models',
        project='current',
        fixed_model_name='nlu'
    )


if __name__ == "__main__":
    train()
