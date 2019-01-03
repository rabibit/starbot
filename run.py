import logging
import os

from rasa_core.broker import PikaProducer
from rasa_core.interpreter import (
    NaturalLanguageInterpreter)
from rasa_core.tracker_store import TrackerStore
from rasa_core.utils import AvailableEndpoints
from rasa_core.run import load_agent, serve_application

logger = logging.getLogger()  # get the root logger


def at_root(p):
    return os.path.abspath(p)


if __name__ == '__main__':
    # Running as standalone python application
    logger.info("Rasa process starting")

    core = at_root('business/models')
    nlu = at_root('nlu/models/current/nlu')
    endpoints = at_root('business/endpoints.yml')
    credentials = at_root('credentials.yml')

    # TODO: remove this ugly fixes
    os.chdir('nlu')

    port = 5002
    connector = None

    _endpoints = AvailableEndpoints.read_endpoints(endpoints)

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

