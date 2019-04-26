#!/usr/bin/env python

import os
import logging
from pathlib import Path
from starbot.nlu.train import train
from starbot.utils.download import download
from tempfile import TemporaryDirectory

BERT_MODEL_URL = "http://cloud.kvin.wang:88/s/ZabQxpnJeHBymg6/download"
BERT_MODEL_FILE = "checkpoint.zip"

logger = logging.getLogger(__name__)
base = Path(__file__).parent


def download_bert_model_if_need():
    if not base.joinpath('checkpoint').exists():
        with TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, BERT_MODEL_FILE)
            logger.info('Downloading {}'.format(BERT_MODEL_URL))
            download(BERT_MODEL_URL, tmpfilename)
            logger.info('Download {} finished'.format(tmpfilename))
            os.system('unzip {}'.format(tmpfilename))


def main():
    mdfile = base / 'data' / 'nlu.md'
    nlu_config = base / 'nlu_config.yml'
    download_bert_model_if_need()
    train(mdfile, nlu_config, base_dir=base, path=base / 'models')


if __name__ == '__main__':
    main()
