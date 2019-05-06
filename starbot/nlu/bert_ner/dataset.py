import typing

if typing.TYPE_CHECKING:
    from typing import List
    from rasa_nlu.training_data.message import Message


class Example:
    def __init__(self, char, label):
        self.char = char
        self.label = label

    def __repr__(self):
        return '<{}:{}>'.format(self.char, self.label)


class Dataset:
    def __init__(self, examples, labels):
        """
        :type examples: List[List[Example]]
        :type labels: List[str]
        """
        self.examples = examples
        self.labels = labels
        self.label_map = {l: i for i, l in enumerate(labels)}
        self.label_reversed = {i: l for i, l in enumerate(labels)}

    def label2id(self, labels):
        assert not isinstance(labels, str)
        return [self.label_map.get(label) for label in labels]


def create_dataset(examples):
    """

    :type examples: List[Message]
    :rtype: Dataset
    """

    dataset = []
    labels = {'O', '[CLS]', '[SEP]'}

    for msg in examples:
        entities = msg.data.get('entities')
        if not entities:
            continue
        sent = [Example(c, 'O') for c in msg.text]
        for entity in entities:
            s = entity['start']
            e = entity['end']
            name = entity['entity']
            sent[s].label = "B-" + name
            for i in range(s+1, e):
                sent[i].label = "I-" + name
            labels.add("B-" + name)
            labels.add("I-" + name)
        sent.insert(0, Example('[CLS]', '[CLS]'))
        sent.append(Example('[SEP]', '[SEP]'))
        dataset.append(sent)
    labels = sorted(labels)
    return Dataset(dataset, labels)


def mark_message_with_labels(message_text, labels):
    entities = []
    name = None
    start = 0

    for i, label in enumerate(labels):
        if label == '[SEP]':
            break
        entity_end = False
        if name and i >= len(message_text):
            entity_end = True
        elif label[0] != 'I' and name:
            entity_end = True
        elif label[0] == 'I':
            name_ = label.split('-', maxsplit=1)[-1]
            if name and name != name_:
                entity_end = True

        if entity_end:
            entities.append({
                'start': start,
                'end': i,
                'entity': name,
                'value': message_text[start:i]
            })
            name = None

        if label[0] in 'BI':
            name = label.split('-', maxsplit=1)[-1]
            start = i
        if i >= len(message_text):
            break
    return entities
