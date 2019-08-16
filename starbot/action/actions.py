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
            events = handler.process(dispatcher, tracker, domain)
            if events is None:
                continue
            logger.debug(f'Handler {handler} processed {tracker.latest_message}')
            return events
        dispatcher.utter_template('utter_default', tracker)
        return []

