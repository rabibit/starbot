from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Any, List

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers import Tokenizer, Token
from rasa.nlu.training_data import Message
from rasa.nlu.training_data import TrainingData


class ThulacNimTokenizer(Tokenizer, Component):
    provides = ["tokens"]
    language_list = ["zh"]

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["thulac_nim"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # type: (Doc) -> List[Token]
        tokens = []
        start = 0

        for tk, _ in self.thulac.cutline(text):
            tokens.append(Token(tk, start))
            start += len(tk)

        return tokens

    @property
    def thulac(self):
        if not hasattr(self, '_thulac'):
            import thulac_nim as thulac
            self._thulac = thulac.Thulac(seg_only=True, prefer_short=True)
        return self._thulac
