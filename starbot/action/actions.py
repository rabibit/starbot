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
        if tracker.latest_message:
            confidence = tracker.latest_message.get('intent', {}).get('confidence')
            if confidence is not None and confidence < 0.99:
                dispatcher.utter_template('utter_default', tracker)
                return []
        for handler in handlers:
            if not handler.match(tracker, domain):
                continue
            events = handler.process(dispatcher, tracker, domain)
            if events is None:
                continue
            logger.debug(f'Handler {handler} processed \u001b[32m{tracker.latest_message}\u001b[0m')
            return events
        dispatcher.utter_template('utter_default', tracker)
        return []

