#!/usr/bin/env python

import os
import sys
import shutil
import logging
import tensorflow as tf
from pathlib import Path
from starbot.nlu.preparemd import convert
from starbot.utils.download import download
from tempfile import TemporaryDirectory

BERT_MODEL_URL = "https://cloud.kvin.wang:8443/s/ZabQxpnJeHBymg6/download"
BERT_MODEL_FILE = "checkpoint.zip"
BERT_MODEL_DIRNAME = "chinese_L-12_H-768_A-12"

MITIE_MODEL_URL = "https://cloud.kvin.wang:8443/s/XEeQkZeqYb7fDYT/download"
MITIE_MODEL_FILE = "total_word_feature_extractor.dat"

logging.getLogger().setLevel("DEBUG")

# This fixes that tensorflow 1.14 don't emit logs
tf.get_logger().addHandler(logging.StreamHandler(sys.stdout))

logger = logging.getLogger(__name__)
base = Path(__file__).parent


def download_bert_model_if_need():
    if not base.joinpath('checkpoint').exists():
        with TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, BERT_MODEL_FILE)
            logger.info('Downloading {}'.format(BERT_MODEL_URL))
            download(BERT_MODEL_URL, tmpfilename)
            logger.info('Download {} finished'.format(tmpfilename))
            os.system('unzip -d "{}" "{}"'.format(base, tmpfilename))
            shutil.move(base/BERT_MODEL_DIRNAME, base/"checkpoint")


def download_mitie_model_if_need():
    if not base.joinpath(MITIE_MODEL_FILE).exists():
        with TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, MITIE_MODEL_FILE)
            logger.info('Downloading {}'.format(MITIE_MODEL_URL))
            download(MITIE_MODEL_URL, tmpfilename)
            logger.info('Download {} finished'.format(tmpfilename))
            shutil.move(tmpfilename, base/MITIE_MODEL_FILE)


def main():
    if len(sys.argv) != 2:
        print("Usage ./train.py bert|mitie")
        sys.exit()
    config = sys.argv[1]

    if config == 'bert':
        nlu_config_file = "bert_nlu_config.yml"
        download_bert_model_if_need()
    elif config == 'mitie':
        nlu_config_file = "mitie_nlu_config.yml"
        download_mitie_model_if_need()
    else:
        print("Usage ./train.py bert|mitie")
        sys.exit()

    mdfile = base/'data'/'nlu.md'
    tmp_nlu_config_file = base/'tmp_nlu_config.yml'

    if not tmp_nlu_config_file.exists():
        shutil.copy(base/'configs'/nlu_config_file, tmp_nlu_config_file)

    convert(mdfile, "rasa_prj/data/nlu.md")

    os.system('cat {} configs/policy_config.yml > rasa_prj/config.yml'.format(tmp_nlu_config_file))
    from rasa.__main__ import main
    os.chdir('rasa_prj')
    os.system('mkdir -p tmp')
    os.environ['TMP'] = 'tmp'
    os.environ['LOG_LEVEL_LIBRARIES'] = 'INFO'
    sys.argv = sys.argv[:1] + ['train']
    main()


if __name__ == '__main__':
    main()
