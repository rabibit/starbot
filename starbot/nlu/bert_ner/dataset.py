import json
import typing

from typing import List, Dict, Iterable
from rasa.nlu.training_data.message import Message


class Example:
    def __init__(self, chars, labels):
        self.chars = chars
        self.labels = labels


class LabelMap:
    labels: List
    map: Dict[str, int]
    reverse_map: Dict[int, str]

    def __init__(self, labels):
        self.labels = sorted(labels)
        self.map = {l: i for i, l in enumerate(self.labels)}
        self.reverse_map = {i: l for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.labels)

    def encode(self, labels: Iterable[str]) -> List[int]:
        return [self.map.get(label) for label in labels]

    def decode(self, label_ids: Iterable[int]) -> List[str]:
        return [self.reverse_map.get(label) for label in label_ids]

    def save(self, filename):
        json.dump(self.labels, open(filename, 'w'))

    @classmethod
    def load(cls, filename) -> 'LabelMap':
        labels = json.load(open(filename))
        return cls(labels)


class Sentence(typing.NamedTuple):
    chars: List[str]
    labels: List[str]
    intent: str


class Dataset:
    examples: List[Sentence]
    ner_labels: LabelMap
    intent_labels: LabelMap

    def __init__(self,
                 examples: Iterable[Sentence],
                 ner_labels: Iterable[str],
                 intent_labels: Iterable[str]):
        self.examples = list(examples)
        self.ner_labels = LabelMap(['[PAD]'] + sorted(ner_labels))
        self.intent_labels = LabelMap(['[PAD]'] + sorted(intent_labels))

    def ner_label2id(self, labels):
        assert not isinstance(labels, str)
        return self.ner_labels.encode(labels)

    def intent_label2id(self, labels):
        assert not isinstance(labels, str)
        return self.intent_labels.encode(labels)

    def intent_label2onehot(self, label):
        cnt = len(self.intent_labels.labels)
        if label == "other":
            return [1.0 / cnt] * cnt
        else:
            label_id = self.intent_labels.encode([label])[0]
            onehot = [0.0] * cnt
            onehot[label_id] = 1.0
            return onehot

    def ood_label2id(self, labels):
        assert not isinstance(labels, str)
        return [1.0 if 'other' == label else 0.0 for label in labels]


def create_dataset(examples: Iterable[Message]) -> Dataset:
    sentences = []
    global_labels = {'O', '[CLS]', '[SEP]'}
    global_intents = set()

    for msg in examples:
        entities = msg.data.get('entities') or []
        intent = msg.data.get('intent')
        global_intents.add(intent)
        chars = list(msg.text)
        labels = ['O' for _ in chars]
        for entity in entities:
            s = entity['start']
            e = entity['end']
            name = entity['entity']
            labels[s] = "B-" + name
            for i in range(s+1, e):
                labels[i] = "I-" + name
            global_labels.add("B-" + name)
            global_labels.add("I-" + name)
        chars = ['[CLS]'] + chars + ['[SEP]']
        labels = ['[CLS]'] + labels + ['[SEP]']
        sentences.append(Sentence(chars=chars, labels=labels, intent=intent))
    return Dataset(sentences, global_labels, global_intents)


def mark_message_with_labels(message_text, labels):
    entities = []
    name = None

def create_dataset(examples: Iterable[Message]) -> Dataset:
    sentences = []
    global_labels = {'O', '[CLS]', '[SEP]'}
    global_intents = set()

    for msg in examples:
        entities = msg.data.get('entities') or []
        intent = msg.data.get('intent')
        global_intents.add(intent)
        chars = list(msg.text)
        labels = ['O' for _ in chars]
        for entity in entities:
            s = entity['start']
            e = entity['end']
            name = entity['entity']
            labels[s] = "B-" + name
            for i in range(s+1, e):
                labels[i] = "I-" + name
            global_labels.add("B-" + name)
            global_labels.add("I-" + name)
        chars = ['[CLS]'] + chars + ['[SEP]']
        labels = ['[CLS]'] + labels + ['[SEP]']
        sentences.append(Sentence(chars=chars, labels=labels, intent=intent))
    return Dataset(sentences, global_labels, global_intents)


def mark_message_with_labels(message_text, labels):
    entities = []
    name = None
    start = 0

    for i, label in enumerate(labels):
        entity_end = False
        if name:
            if label == '[SEP]' or i >= len(message_text) or label[0] != 'I':
                entity_end = True
            elif label[0] == 'I':
                if name != label.split('-', maxsplit=1)[-1]:
                    entity_end = True

        if entity_end:
            entities.append({
                'start': start,
                'end': i,
                'entity': name,
                'value': message_text[start:i]
            })
            name = None

        if label[0] in 'B' or label[0] in 'I' and not name:
            name = label.split('-', maxsplit=1)[-1]
            start = i
        if i >= len(message_text):
            break
    return entities
