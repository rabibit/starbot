#!/usr/bin/env python

from starbot.nlu import preparemd

from rasa_nlu.train import train as rasa_train
from rasa_nlu import config, utils
from pathlib import Path
import tempfile
import logging
import os


def train(datafile: str, nlu_config: str, base_dir: str, path='models', project='current', model_name='nlu'):
    #utils.configure_colored_logging(logging.INFO)
    nlu_config = config.load(nlu_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        nlu_config.override({
            "base_dir": base_dir,
            "tmp_dir": tmpdir
        })
        tmpmd = os.path.join(tmpdir, "tmp-nlu.md")
        preparemd.convert(datafile, tmpmd, repeat=1)
        rasa_train(
            nlu_config=nlu_config,
            data=tmpmd,
            path=path,
            project=project,
            fixed_model_name=model_name
        )


if __name__ == "__main__":
    import starbot
    base = Path(starbot.__file__).parent.parent / 'run'
    mdfile = base / 'data' / 'nlu.md'
    nlu_config = base / 'nlu_config.yml'
    train(str(mdfile), nlu_config, base_dir=str(base), path=base / 'models')
