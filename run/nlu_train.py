#!/usr/bin/env python

import os
import shutil
import logging
from pathlib import Path
from starbot.nlu.train import train
from starbot.utils.download import download
from tempfile import TemporaryDirectory

BERT_MODEL_URL = "https://cloud.kvin.wang:8443/s/ZabQxpnJeHBymg6/download"
BERT_MODEL_FILE = "checkpoint.zip"
BERT_MODEL_DIRNAME = "chinese_L-12_H-768_A-12"

MITIE_MODEL_URL = "https://cloud.kvin.wang:8443/s/XEeQkZeqYb7fDYT/download"
MITIE_MODEL_FILE = "total_word_feature_extractor.dat"

#CONFIG = "nlu_config.yml"
CONFIG = "rasa_nlu_config.yml"

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
            shutil.move(base / BERT_MODEL_DIRNAME, base / "checkpoint")


def download_mitie_model_if_need():
    if not base.joinpath(MITIE_MODEL_FILE).exists():
        with TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, MITIE_MODEL_FILE)
            logger.info('Downloading {}'.format(MITIE_MODEL_URL))
            download(MITIE_MODEL_URL, tmpfilename)
            logger.info('Download {} finished'.format(tmpfilename))
            shutil.move(tmpfilename, base / MITIE_MODEL_FILE)

def main():
    mdfile = base / 'data' / 'nlu.md'
    nlu_config = base / CONFIG
    if CONFIG == 'nlu_config.yml':
        download_bert_model_if_need()
    elif CONFIG == 'rasa_nlu_config.yml':
        download_mitie_model_if_need()

    train(mdfile, nlu_config, base_dir=base, path=base / 'models')


if __name__ == '__main__':
    main()
