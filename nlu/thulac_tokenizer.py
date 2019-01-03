from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Any, List

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData


class ThulacTokenizer(Tokenizer, Component):
    name = "tokenizer_thulac"

    provides = ["tokens"]

    language_list = ["zh"]

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["thulac"]

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

        for tk, _ in self.thulac.cut(text):
            tokens.append(Token(tk, start))
            start += len(tk)

        return tokens

    @property
    def thulac(self):
        if not hasattr(self, '_thulac'):
            import thulac
            self._thulac = thulac.thulac()
        return self._thulac
