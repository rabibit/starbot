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


cixing = ['uw', 'i', 'ni', 'id', 'j', 'f', 'x', 'u', 'q', 'm', 'g', 'r', 'p', 'o', 'v', 'd', 's', 'c', 'n', 'k', 'ns', 'w', 'e', 'np', 'a', 'nz', 'h', 't']
num_cixing = len(cixing)
cixing_map = {}
for i in range(len(cixing)):
    cixing_map[cixing[i]] = i


fenci = ['B', 'I', 'S']
num_fenci = len(fenci)
fenci_map = {}
for i in range(len(fenci)):
    fenci_map[fenci[i]] = i


class ThulacTokenizer(Tokenizer, Component):
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
        # type: (Doc) -> List[List[int]]
        tokens = []

        for a, b in self.thulac.cut(text):
            if len(a) > 1:
                fenci_vec = [0]*(num_cixing + num_fenci)
                fenci_vec[fenci_map['B']] = 1
                fenci_vec[cixing_map[b] + num_fenci] = 1
                tokens.append(fenci_vec)
                for sub_a in a[1:]:
                    fenci_vec = [0] * (num_cixing + num_fenci)
                    fenci_vec[fenci_map['I']] = 1
                    fenci_vec[cixing_map[b] + num_fenci] = 1
                    tokens.append(fenci_vec)
            else:
                fenci_vec = [0]*(num_cixing + num_fenci)
                fenci_vec[fenci_map['B']] = 1
                fenci_vec[cixing_map[b] + num_fenci] = 1
                tokens.append(fenci_vec)

        return tokens

    @property
    def thulac(self):
        if not hasattr(self, '_thulac'):
            import thulac
            self._thulac = thulac.thulac(user_dict='/codes/starbot/run/mydict', seg_only=False)
            self._thulac._thulac__userDict.tag = 'n'  # chang uw to n
        return self._thulac
