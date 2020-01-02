#!/usr/bin/env python

from rasa.nlu.training_data.formats.markdown import MarkdownReader, MarkdownWriter
from rasa.nlu.training_data import Message
from collections import defaultdict
import re


def intent(self):
    return self.data['intent']


Message.intent = property(intent)


ent_regex = re.compile(
    r"\[(?P<entity_text>[^\]]+)" r"\]\((?P<entity>[^:)]*?)" r"(?:\:(?P<value>[^)]+))?\)"
)


def _parse_training_example(self, example):
    """Extract entities and synonyms, and convert to plain text."""
    from rasa.nlu.training_data import Message

    entities = self._find_entities_in_training_example(example)
    plain_text = re.sub(ent_regex, lambda m: m.groupdict()["entity_text"], example)
    self._add_synonyms(plain_text, entities)
    message = Message(plain_text, {"intent": self.current_title})
    if len(entities) > 0:
        modify_infos = []
        for i, entity in enumerate(entities):
            if entity['entity'].startswith('!'):
                tmp = entity.copy()
                tmp['bak'] = entity['entity']
                entities[i]['entity'] = entity['entity'][1:]
                tmp['entity'] = 'wrong'
                modify_infos.append(tmp)
        if len(modify_infos) > 0:
            message.set("modify_info", modify_infos)
        message.set("entities", entities)
    return message


MarkdownReader._parse_training_example = _parse_training_example


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
        msg.entity_space = self.entity_values.copy()
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
                self.entity_values[key.strip()] = [v.strip() for v in values]
            return

        if line.startswith("?"):
            if self.current_section == "intent":
                self.questions.append(line[1:].strip())
            return

        return super(MDReader, self)._parse_item(line)


INTENT = "intent"
SYNONYM = "synonym"
REGEX = "regex"
LOOKUP = "lookup"
available_sections = [INTENT, SYNONYM, REGEX, LOOKUP]


class MDWriter(MarkdownWriter):

    def __init__(self, repeat=1):
        self.repeat = repeat

    def dumps(self, training_data):
        """Transforms a TrainingData object into a markdown string."""
        md = u''
        md += self._generate_training_examples_md(training_data)
        md += self._generate_synonyms_md(training_data)
        md += self._generate_regex_features_md(training_data)
        md += self._generate_lookup_tables_md(training_data)

        return md

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
        if 'true' in message.entity_space.get('_expand_entities', ''):
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
        modify_infos = sorted(message.get('modify_info', []),
                          key=lambda k: k['start'])
        if entities:
            for modify_info in modify_infos:
                for entity in entities:
                    if entity['value'] == modify_info['value']:
                        entity['entity'] = modify_info['bak']

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

    def _generate_synonyms_md(self, training_data):
        """generates markdown for entity synomyms."""
        entity_synonyms = sorted(training_data.entity_synonyms.items(),
                                 key=lambda x: x[1])
        md = u''
        for i, synonym in enumerate(entity_synonyms):
            if i == 0 or entity_synonyms[i - 1][1] != synonym[1]:
                md += self._generate_section_header_md(SYNONYM, synonym[1])

            md += self._generate_item_md(synonym[0])

        return md

    def _generate_regex_features_md(self, training_data):
        """generates markdown for regex features."""
        md = u''
        # regex features are already sorted
        regex_features = training_data.regex_features
        for i, regex_feature in enumerate(regex_features):
            if i == 0 or regex_features[i - 1]["name"] != regex_feature["name"]:
                md += self._generate_section_header_md(REGEX,
                                                       regex_feature["name"])

            md += self._generate_item_md(regex_feature["pattern"])

        return md

    def _generate_lookup_tables_md(self, training_data):
        """generates markdown for regex features."""
        md = u''
        # regex features are already sorted
        lookup_tables = training_data.lookup_tables
        for i, lookup_table in enumerate(lookup_tables):
            md += self._generate_section_header_md(LOOKUP, lookup_table["name"])
            elements = lookup_table["elements"]
            if isinstance(elements, list):
                for e in elements:
                    md += self._generate_item_md(e)
            else:
                md += self._generate_fname_md(elements)
        return md

    @staticmethod
    def _generate_section_header_md(section_type, title,
                                    prepend_newline=True):
        """generates markdown section header."""
        prefix = "\n" if prepend_newline else ""
        return prefix + "## {}:{}\n".format(section_type, title)

    @staticmethod
    def _generate_item_md(text):
        """generates markdown for a list item."""
        return "- {}\n".format(text)

    @staticmethod
    def _generate_fname_md(text):
        """generates markdown for a lookup table file path."""
        return "  {}\n".format(text)

    def _generate_message_md(self, message):
        """generates markdown for a message object."""
        md = ''
        text = message.get('text', "")
        entities = sorted(message.get('entities', []),
                          key=lambda k: k['start'])
        modify_infos = sorted(message.get('modify_info', []),
                              key=lambda k: k['start'])
        if entities:
            for modify_info in modify_infos:
                for entity in entities:
                    if entity['value'] == modify_info['value']:
                        entity['entity'] = modify_info['bak']

        pos = 0
        for entity in entities:
            md += text[pos:entity['start']]
            md += self._generate_entity_md(text, entity)
            pos = entity['end']

        md += text[pos:]

        return md

    @staticmethod
    def _generate_entity_md(text, entity):
        """generates markdown for an entity object."""
        entity_text = text[entity['start']:entity['end']]
        entity_type = entity['entity']
        if entity_text != entity['value']:
            # add synonym suffix
            entity_type += ":{}".format(entity['value'])

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

