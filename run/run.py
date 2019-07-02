#!/usr/bin/env python

import logging
import sys
import os

from rasa.core.processor import MessageProcessor, UserMessage, DialogueStateTracker
from rasa.core.events import BotUttered

logger = logging.getLogger()  # get the root logger


def at_root(p):
    return os.path.abspath(p)


def preprocess(message, tracker):
    def is_question(message):
        return '?' in message.text or '？' in message.text

    message.origin_text = message.text
    if tracker:
        q = tracker.latest_bot_utterance
        if isinstance(q, BotUttered) and q.text and is_question(q):
            message.text = q.text + message.text
    return message


class StarMessageProcessor(MessageProcessor):
    async def _handle_message_with_tracker(
        self, message: UserMessage, tracker: DialogueStateTracker
    ):
        message = preprocess(message, tracker)
        return super(StarMessageProcessor, self)._handle_message_with_tracker(message, tracker)

    async def _parse_message(self, message):
        parsed_data = await super(StarMessageProcessor, self)._parse_message(message)
        parsed_data['text'] = message.origin_text
        return parsed_data


def patch_it():
    from rasa.core import processor
    processor.MessageProcessor = StarMessageProcessor

    from rasa.core import agent
    agent.MessageProcessor = StarMessageProcessor


if __name__ == '__main__':
    patch_it()

    from rasa.__main__ import main
    sys.argv = sys.argv[:1] + ['run']
    main()

