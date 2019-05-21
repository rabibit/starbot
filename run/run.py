#!/usr/bin/env python

import logging
import os
import sys

sys.path.append(os.path.abspath('nlu'))

from rasa_core.broker import PikaProducer
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.tracker_store import TrackerStore
from rasa_core import utils as rsutils
from rasa_core.run import serve_application
from rasa_core.agent import Agent
from rasa_core.processor import MessageProcessor
from rasa_core.events import BotUttered

logger = logging.getLogger()  # get the root logger


def at_root(p):
    return os.path.abspath(p)


class StarMessageProcessor(MessageProcessor):
    def log_message(self, message):
        # type: (UserMessage) -> Optional[DialogueStateTracker]

        # we have a Tracker instance for each user
        # which maintains conversation state
        tracker = self._get_tracker(message.sender_id)

        # preprocess message if necessary
        if self.message_preprocessor is not None:
            message = self.message_preprocessor(message, tracker)

        if tracker:
            self._handle_message_with_tracker(message, tracker)
            # save tracker state to continue conversation from this state
            self._save_tracker(tracker)
        else:
            logger.warning("Failed to retrieve or create tracker for sender "
                           "'{}'.".format(message.sender_id))
        return tracker

    @staticmethod
    def preprocessor(message, tracker):
        def is_question(message):
            return '?' in message.text or '？' in message.text

        message.origin_text = message.text
        if tracker:
            q = tracker.latest_bot_utterance
            if isinstance(q, BotUttered) and q.text and is_question(q):
                message.text = q.text + message.text
        return message

    def _parse_message(self, message):
        parsed_data = super(StarMessageProcessor, self)._parse_message(message)
        parsed_data['text'] = message.origin_text
        return parsed_data


class StarAgent(Agent):
    def create_processor(self, preprocessor=None):
        self._ensure_agent_is_ready()
        mp = StarMessageProcessor(
            self.interpreter,
            self.policy_ensemble,
            self.domain,
            self.tracker_store,
            self.nlg,
            action_endpoint=self.action_endpoint,
            message_preprocessor=preprocessor)
        if preprocessor is None:
            mp.message_preprocessor = mp.preprocessor
        return mp


def load_agent(core_model, interpreter, endpoints,
               tracker_store=None):
    if endpoints.model:
        raise NotImplementedError
    else:
        return StarAgent.load(core_model,
                              interpreter=interpreter,
                              generator=endpoints.nlg,
                              tracker_store=tracker_store,
                              action_endpoint=endpoints.action)


if __name__ == '__main__':
    # Running as standalone python application
    rsutils.configure_colored_logging(logging.DEBUG)

    logger.info("Rasa process starting")

    core = at_root('../starbot/policy/models')
    nlu = at_root('models/current/nlu')
    endpoints = at_root('../starbot/policy/endpoints.yml')
    credentials = at_root('configs/credentials.yml')

    port = 5002
    connector = None

    _endpoints = rsutils.AvailableEndpoints.read_endpoints(endpoints)

    _interpreter = NaturalLanguageInterpreter.create(nlu,
                                                     _endpoints.nlu)
    _broker = PikaProducer.from_endpoint_config(_endpoints.event_broker)

    _tracker_store = TrackerStore.find_tracker_store(
        None, _endpoints.tracker_store, _broker)

    _agent = load_agent(core,
                        interpreter=_interpreter,
                        tracker_store=_tracker_store,
                        endpoints=_endpoints)

    serve_application(_agent, connector, port, credentials)
