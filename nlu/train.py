#!/usr/bin/env python

import extpipelines

from rasa_nlu.train import do_train
from rasa_nlu import config, utils
import logging

if __name__ == "__main__":
    utils.configure_colored_logging(logging.INFO)
    do_train(
        cfg=config.load('nlu_config.yml'),
        data='nlu-gen.md',
        path='models',
        project='current',
        fixed_model_name='nlu'
    )
