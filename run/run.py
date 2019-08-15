#!/usr/bin/env python

import shutil
import logging
import sys
import os

os.environ['TMP'] = 'tmp'
os.system('rm -rf tmp')
os.mkdir('tmp')

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
    add_prequestion = False

    async def _handle_message_with_tracker(
        self, message: UserMessage, tracker: DialogueStateTracker
    ):
        if self.add_prequestion:
            message = preprocess(message, tracker)
        return await super(StarMessageProcessor, self)._handle_message_with_tracker(message, tracker)

    async def _parse_message(self, message, *a):
        parsed_data = await super(StarMessageProcessor, self)._parse_message(message, *a)
        if self.add_prequestion:
            parsed_data['text'] = message.origin_text
        logger.info("parsed data: {}".format(parsed_data))
        return parsed_data


def patch_it():
    from rasa.core import processor
    processor.MessageProcessor = StarMessageProcessor

    from rasa.core import agent
    agent.MessageProcessor = StarMessageProcessor

    from rasa import model

    origin_create_package_rasa = model.create_package_rasa

    def create_package_rasa(training_directory, output_filename, fingerprint):
        outdir = os.path.dirname(output_filename)
        outdir_models = os.path.join(outdir, 'models')
        shutil.rmtree(outdir_models, ignore_errors=True)
        os.makedirs(outdir_models, exist_ok=True)

        print('coping models to {}'.format(outdir_models))
        shutil.copytree(training_directory, outdir_models, symlinks=True)
        return origin_create_package_rasa(training_directory, output_filename, fingerprint)

    model.create_package_rasa = create_package_rasa

    os.chdir('rasa_prj')

if __name__ == '__main__':
    patch_it()

    from rasa.__main__ import main
    sys.argv = sys.argv[:1] + ['run', '--enable-api']
    main()

