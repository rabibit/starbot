
from typing import Optional, Dict, Text, Any
from rasa.nlu.components import Component, TrainingData, RasaNLUModelConfig, Message, Metadata
from pathlib import Path
import json


class CharFreqClassifier(Component):

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)
        self.all_chars = set()

    @staticmethod
    def get_model_path(model_dir: Text):
        pdir = Path(model_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        return pdir / 'allchars.json'

    @classmethod
    def load(
            cls,
            meta: Dict[Text, Any],
            model_dir: Optional[Text] = None,
            model_metadata: Optional["Metadata"] = None,
            cached_component: Optional["Component"] = None,
            **kwargs: Any
    ) -> "Component":
        self: CharFreqClassifier = super(CharFreqClassifier, cls).load(
            meta, model_dir, model_metadata, cached_component, **kwargs)
        filename = self.get_model_path(model_dir)
        self.all_chars = set(json.load(open(filename)))
        return self

    def train(
            self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs: Any
    ) -> None:
        for message in training_data.training_examples:
            message: Message = message
            self.all_chars.update(set(message.text))

    def process(self, message: Message, **kwargs: Any) -> None:
        pass

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        filename = self.get_model_path(model_dir)
        json.dump(list(self.all_chars), open(filename, 'wb'))
        return None
