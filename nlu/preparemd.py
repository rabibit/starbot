#!/usr/bin/env python

from rasa_nlu.training_data.formats.markdown import MarkdownReader, MarkdownWriter
from collections import defaultdict


class MDReader(MarkdownReader):
    def __init__(self):
        super(MDReader, self).__init__()
        self.entity_values = defaultdict(list)

    def _parse_item(self, line):
        if line.startswith("@"):
            key, values = line.strip()[1:].split(":")
            values = values.split('|')
            self.entity_values[key.strip()].extend([v.strip() for v in values])
            return
        return super(MDReader, self)._parse_item(line)


class MDWriter(MarkdownWriter):
    def __init__(self, entity_values):
        self.entity_values = entity_values

    def _generate_message_md(self, message):
        """generates markdown for a message object."""
        md = super(MDWriter, self)._generate_message_md(message)
        md += self._generate_extra_msg_md(message)
        return md

    def _generate_extra_msg_md(self, message):
        found = [False]
        entities = sorted(message.get('entities', []),
                          key=lambda k: k['start'])

        def yield_msgs(msg, prefix, pos, entities):
            if not entities:
                yield prefix + msg['text'][pos:]
                return
            entity = entities[0]
            prefix += msg['text'][pos:entity['start']]
            space = self.entity_values.get(entity['entity'], [])
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

        md = ''
        for md_ in yield_msgs(message, '', 0, entities):
            md += '\n- ' + md_

        return md if found[0] else ''

    def _generate_entity_md_with_replace(self, text, entity):
        """generates markdown for an entity object."""
        entity_text = text
        entity_type = entity['entity']
        return '[{}]({})'.format(entity_text, entity_type)


if __name__ == '__main__':
    reader = MDReader()
    data = reader.read("nlu.md")
    MDWriter(reader.entity_values).dump('nlu-gen.md', data)



