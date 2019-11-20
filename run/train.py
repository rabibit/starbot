#!/usr/bin/env python

import os
import sys
import shutil
import logging
import argparse

tmpdir = os.path.abspath('.tmp')
os.environ['TMP'] = tmpdir
shutil.rmtree(tmpdir, ignore_errors=True)
os.makedirs(tmpdir, exist_ok=True)

from pathlib import Path
from tempfile import TemporaryDirectory

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--model', default='bert', choices=['bert', 'mitie'], help='Using which model, bert or mitie')
parser.add_argument('-m', '--module', default='all', choices=['all', 'core', 'nlu'], help='The module to be trained')

BERT_MODEL_URL = "https://cloud.kvin.wang:8443/s/ZabQxpnJeHBymg6/download"
BERT_MODEL_FILE = "checkpoint.zip"
BERT_MODEL_DIRNAME = "chinese_L-12_H-768_A-12"

MITIE_MODEL_URL = "https://cloud.kvin.wang:8443/s/XEeQkZeqYb7fDYT/download"
MITIE_MODEL_FILE = "total_word_feature_extractor.dat"

logging.getLogger().setLevel("DEBUG")

logger = logging.getLogger(__name__)
base = Path(__file__).parent


def download_bert_model_if_need():
    from starbot.utils.download import download
    if not base.joinpath('checkpoint').exists():
        with TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, BERT_MODEL_FILE)
            logger.info('Downloading {}'.format(BERT_MODEL_URL))
            download(BERT_MODEL_URL, tmpfilename)
            logger.info('Download {} finished'.format(tmpfilename))
            os.system('unzip -d "{}" "{}"'.format(base, tmpfilename))
            shutil.move(base/BERT_MODEL_DIRNAME, base/"checkpoint")


def download_mitie_model_if_need():
    from starbot.utils.download import download
    if not base.joinpath(MITIE_MODEL_FILE).exists():
        with TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, MITIE_MODEL_FILE)
            logger.info('Downloading {}'.format(MITIE_MODEL_URL))
            download(MITIE_MODEL_URL, tmpfilename)
            logger.info('Download {} finished'.format(tmpfilename))
            shutil.move(tmpfilename, base/MITIE_MODEL_FILE)


def config_tf_log():
    import tensorflow as tf
    # This fixes that tensorflow 1.14 don't emit logs
    tf.get_logger().addHandler(logging.StreamHandler(sys.stdout))

def patch_rasa_for_tf2():
    import tensorflow as tf
    import sys
    tf.logging = tf.compat.v1.logging
    tf.ConfigProto = None

    sys.modules['rasa.core.policies.embedding_policy'] = type(sys)("embedding_policy")
    sys.modules['rasa.core.policies.embedding_policy'].EmbeddingPolicy = None
    sys.modules['rasa.core.policies.keras_policy'] = type(sys)("embedding_policy")
    sys.modules['rasa.core.policies.keras_policy'].KerasPolicy = None

    class EmbeddingIntentClassifier:
        name = "EmbeddingIntentClassifier"

    sys.modules['rasa.nlu.classifiers.embedding_intent_classifier'] = type(sys)("")
    sys.modules['rasa.nlu.classifiers.embedding_intent_classifier'].EmbeddingIntentClassifier = EmbeddingIntentClassifier

def main():
    args = parser.parse_args()
    config = args.model

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
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

    from starbot.nlu.preparemd import convert
    convert(mdfile, "rasa_prj/data/nlu.md")

    os.system('cat {} configs/policy_config.yml > rasa_prj/config.yml'.format(tmp_nlu_config_file))
    from rasa.__main__ import main
    os.chdir('rasa_prj')
    os.environ['LOG_LEVEL_LIBRARIES'] = 'INFO'
    sys.argv = sys.argv[:1] + ['train']
    if args.module != 'all':
        sys.argv.append(args.module)

    patch_rasa()
    patch_rasa_for_tf2()
    config_tf_log()
    main()


def patch_rasa():
    from typing import Text, Optional
    from rasa import model
    from rasa.model import persist_fingerprint, Fingerprint

    def create_package_rasa(
            training_directory: Text,
            output_filename: Text,
            fingerprint: Optional[Fingerprint] = None,
    ) -> Text:
        """Creates a zipped Rasa model from trained model files.

        Args:
            training_directory: Path to the directory which contains the trained
                                model files.
            output_filename: Name of the zipped model file to be created.
            fingerprint: A unique fingerprint to identify the model version.

        Returns:
            Path to zipped model.

        """
        import tarfile

        if fingerprint:
            persist_fingerprint(training_directory, fingerprint)

        output_directory = os.path.dirname(output_filename)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        print(f'output: {output_filename}')

        with tarfile.open(output_filename, "w") as tar:
            for elem in os.scandir(training_directory):
                tar.add(elem.path, arcname=elem.name)
                print(f'add {elem.path}')

        shutil.rmtree(training_directory)
        return output_filename

    model.create_package_rasa = create_package_rasa


if __name__ == '__main__':
    main()

