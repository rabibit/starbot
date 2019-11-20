#!/usr/bin/env python

import shutil
import logging
import sys
import os
import tensorflow as tf

os.environ['TMP'] = 'tmp'
os.system('rm -rf tmp')
os.mkdir('tmp')


def patch_rasa_for_tf2():
    import tensorflow as tf
    import sys
    tf.logging = tf.compat.v1.logging
    tf.ConfigProto = None

    sys.modules['rasa.core.policies.embedding_policy'] = type(sys)("embedding_policy")
    sys.modules['rasa.core.policies.embedding_policy'].EmbeddingPolicy = None
    sys.modules['rasa.core.policies.keras_policy'] = type(sys)("embedding_policy")
    sys.modules['rasa.core.policies.keras_policy'].KerasPolicy = None

    class EmbeddingIntentClassifier:
        name = "EmbeddingIntentClassifier"

    sys.modules['rasa.nlu.classifiers.embedding_intent_classifier'] = type(sys)("")
    sys.modules['rasa.nlu.classifiers.embedding_intent_classifier'].EmbeddingIntentClassifier = EmbeddingIntentClassifier


def patch_rasa():
    from typing import Text, Optional
    from rasa import model
    from rasa.model import persist_fingerprint, Fingerprint

    def create_package_rasa(
            training_directory: Text,
            output_filename: Text,
            fingerprint: Optional[Fingerprint] = None,
    ) -> Text:
        """Creates a zipped Rasa model from trained model files.

        Args:
            training_directory: Path to the directory which contains the trained
                                model files.
            output_filename: Name of the zipped model file to be created.
            fingerprint: A unique fingerprint to identify the model version.

        Returns:
            Path to zipped model.

        """
        import tarfile

        if fingerprint:
            persist_fingerprint(training_directory, fingerprint)

        output_directory = os.path.dirname(output_filename)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        print(f'output: {output_filename}')

        with tarfile.open(output_filename, "w") as tar:
            for elem in os.scandir(training_directory):
                tar.add(elem.path, arcname=elem.name)
                print(f'add {elem.path}')

        shutil.rmtree(training_directory)
        return output_filename

    model.create_package_rasa = create_package_rasa
patch_rasa()
patch_rasa_for_tf2()
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

    os.chdir('rasa_prj')


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    patch_it()

    from rasa.__main__ import main
    sys.argv = sys.argv[:1] + ['run', '--enable-api', '--debug', ]
    main()

