from typing import Any

from rasa.nlu.components import Component
from rasa.nlu.training_data import Message


class ClearEmbedding(Component):
    requires = ['bert_embedding']

    def process(self, message: Message, **kwargs: Any) -> None:
        message.set("bert_embedding", None)
