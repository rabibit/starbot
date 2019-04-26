#!/usr/bin/env python

from rasa_nlu.training_data.formats.markdown import MarkdownReader, MarkdownWriter
from rasa_nlu.training_data import Message
from collections import defaultdict


def intent(self):
    return self.data['intent']


Message.intent = property(intent)


class MDReader(MarkdownReader):
    def __init__(self):
        super(MDReader, self).__init__()
        self.entity_values = defaultdict(list)
        self.questions = []

    def _set_current_section(self, section, title):
        super(MDReader, self)._set_current_section(section, title)
        self.questions = []

    def _parse_training_example(self, example):
        msg = super(MDReader, self)._parse_training_example(example)
        msg.questions = self.questions
        msg.entity_space = self.entity_values
        return msg

    def _parse_item(self, line):
        line = line.strip()
        if line.startswith("@"):
            key, values = line.strip()[1:].split(":")
            values = values.strip()
            if values[:1] == "=":
                self.entity_values[key.strip()] = self.entity_values[values[1:].strip()]
            else:
                values = values.split('|')
                self.entity_values[key.strip()].extend([v.strip() for v in values])
            return

        if line.startswith("?"):
            if self.current_section == "intent":
                self.questions.append(line[1:].strip())
            return

        return super(MDReader, self)._parse_item(line)


class MDWriter(MarkdownWriter):
    def __init__(self, repeat=1):
        self.repeat = repeat

    def _generate_training_examples_md(self, training_data):
        """generates markdown training examples."""
        training_examples = sorted([e for e in training_data.training_examples],
                                   key=lambda k: k.intent)
        md = u''
        for i, example in enumerate(training_examples):
            if i == 0 or training_examples[i - 1].intent != example.intent:
                md += self._generate_section_header_md("intent", example.intent, i != 0)

            md += self._generate_item_md(self._generate_message_md2(example))

        return md

    def _generate_message_md2(self, message):
        """generates markdown for a message object."""
        items = [self._generate_message_md(message.as_dict())]
        for item in self._generate_extra_msg_md(message):
            items.append(item)
        for md in items[:]:
            for q in message.questions:
                items.append(q + md)
        items *= self.repeat
        return '\n- '.join(items)

    def _generate_extra_msg_md(self, msgobj):
        found = [False]
        message = msgobj.as_dict()
        entities = sorted(message.get('entities', []),
                          key=lambda k: k['start'])

        def yield_msgs(msg, prefix, pos, entities):
            if not entities:
                yield prefix + msg['text'][pos:]
                return
            entity = entities[0]
            prefix += msg['text'][pos:entity['start']]
            space = msgobj.entity_space.get(entity['entity'], [])
            pos = entity['end']
            if not space:
                prefix += self._generate_entity_md(msg['text'], entity)
                for m in yield_msgs(msg, prefix, pos, entities[1:]):
                    yield m
            else:
                for v in space:
                    found[0] = True
                    p0 = prefix + self._generate_entity_md_with_replace(v, entity)
                    for m in yield_msgs(msg, p0, pos, entities[1:]):
                        yield m

        items = []
        for md in yield_msgs(message, '', 0, entities):
            items.append(md)

        return items if found[0] else []

    def _generate_entity_md_with_replace(self, text, entity):
        """generates markdown for an entity object."""
        entity_text = text
        entity_type = entity['entity']
        return '[{}]({})'.format(entity_text, entity_type)


def converts(infilename, repeat=1):
    reader = MDReader()
    data = reader.read(infilename)
    return MDWriter(repeat=repeat).dumps(data)


def convert(infilename, outfilename, repeat=1):
    open(outfilename, 'w').write(converts(infilename, repeat))


if __name__ == '__main__':
    convert("nlu.md", "tmp-nlu.md")
    # print(converts("nlu.md"))

