import re

from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.training_data import Message


class CommandExtractor(EntityExtractor):
    """

    >>> m = Message("不要了")
    >>> CommandExtractor().process(m)
    >>> m.get('commands')
    ['no-more']

    >>> m = Message("就这些啦")
    >>> CommandExtractor().process(m)
    >>> m.get('commands')
    ['no-more']

    >>> m = Message("就要这些")
    >>> CommandExtractor().process(m)
    >>> m.get('commands')
    ['no-more']

    >>> m = Message("退出")
    >>> CommandExtractor().process(m)
    >>> m.get('commands')
    ['exit']

    >>> m = Message("你说啥")
    >>> CommandExtractor().process(m)
    >>> m.get('commands')
    ['what?']

    >>> m = Message("啥")
    >>> CommandExtractor().process(m)
    >>> m.get('commands')
    ['what?']

    >>> m = Message("啥时候退房")
    >>> CommandExtractor().process(m)
    >>> m.get('commands')
    []
    """

    provides = ['commands']

    full_matches = {
        'no-more': {'不要了', '不用了', '够了'},
        'exit': {'退出'},
        'cancel': {'取消', '不买了', '返回'},
    }

    re_matches = {
        'no-more': re.compile(r"就要?这些|就这样|没其[他它]了|不要其[他它]了"),
        'what?': re.compile(r"你说啥|^啥$|没听清|你说什么|^什么$|再说一|重复一"),
    }

    def process(self, message: Message, **kwargs) -> None:
        commands = set()
        for k, s in self.full_matches.items():
            # TODO: strip off punctuations
            if message.text in s:
                commands.add(k)
        for k, pat in self.re_matches.items():
            if pat.search(message.text):
                commands.add(k)
        message.set('commands', list(commands), add_to_output=True)
