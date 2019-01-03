#!/usr/bin/env python

#python -m rasa_nlu.train -c nlu_config.yml --data nlu.md -o models --fixed_model_name nlu --project current --verbose
from thulac_tokenizer import ThulacTokenizer

from rasa_nlu.registry import registered_components

registered_components[ThulacTokenizer.name] = ThulacTokenizer

from rasa_nlu.train import do_train
from rasa_nlu import config, utils
import logging

if __name__ == "__main__":
    utils.configure_colored_logging(logging.INFO)
    do_train(
        cfg=config.load('nlu_config.yml'),
        data='nlu.md',
        path='models',
        project='current',
        fixed_model_name='nlu'
    )
