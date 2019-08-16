import logging
from rasa_sdk import Action

from typing import Text, Dict, Any, List
from rasa_sdk.executor import CollectingDispatcher, Tracker
from .intent_handlers import handlers

logger = logging.getLogger(__name__)


class ProcessIntentAction(Action):
    def name(self):
        return "action_process_intent"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        for handler in handlers:
            if handler.match(tracker, domain):
                logger.debug(f'Matched handler {handler} for {tracker.latest_message}')
                rv = handler.process(dispatcher, tracker, domain)
                return rv or []
        dispatcher.utter_template('utter_default', tracker)
        return []

