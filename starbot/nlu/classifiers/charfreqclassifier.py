
from typing import Optional, Dict, Text, Any
from rasa.nlu.components import Component, TrainingData, RasaNLUModelConfig, Message
from rasa.nlu.model import Metadata
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
            if message.data['intent'] != 'other':
                self.all_chars.update(set(message.text))

    def process(self, message: Message, **kwargs: Any) -> None:
        ir = message.get("intent")
        if '?' in message.text:
            original_text = message.text[message.text.index('?') + 1:]
        else:
            original_text = message.text

        in_vocab_words = [word
                            for word in original_text
                                if word in self.all_chars]
        print('rate: %s' % (len(in_vocab_words) / len(original_text)))
        print('words: %s' % in_vocab_words)
        if (len(in_vocab_words) / len(original_text)) < 0.6:
            ir['confidence'] = 0
        if '没有问题' in message.text:
            ir['name'] = 'confirm'
            ir['confidence'] = 1
            message.set("intent", 'confirm', add_to_output=True)
        message.set("intent", ir, add_to_output=True)

        return None

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        filename = self.get_model_path(model_dir)
        json.dump(list(self.all_chars), open(filename, 'w'))
        return None
