
import os
import logging
from typing import Any

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.training_data import load_data

from starbot.nlu.preparemd import convert

logger = logging.getLogger(__name__)


class PrepareDataset(Component):
    """
    """

    def train(
            self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs: Any
    ) -> None:
        train_mdfile = self.component_config.get("train")
        eval_mdfile = self.component_config.get("eval")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            def load_data_ex(filename):
                tmp_filename = os.path.join(tmpdir, os.path.basename(filename))
                convert(filename, tmp_filename)
                logger.info(f'Loading dataset: {tmp_filename}')
                return load_data(tmp_filename)
            training_data.training_examples = load_data_ex(train_mdfile).training_examples
            training_data.eval_examples = load_data_ex(eval_mdfile).training_examples

