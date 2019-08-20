import logging
from rasa_sdk import Action

from typing import Text, Dict, Any, List
from rasa_sdk.executor import CollectingDispatcher, Tracker
from .intent_handlers import handlers
import random

logger = logging.getLogger(__name__)


class ProcessIntentAction(Action):
    def name(self):
        return "action_process_intent"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        if tracker.latest_message:
            if handlers[0].is_last_message_user(tracker):
                dispatcher.utter_message(f'/intent is {tracker.latest_message.get("intent")}')
                dispatcher.utter_message(f'/entities {tracker.latest_message.get("entities")}')
            confidence = tracker.latest_message.get('intent', {}).get('confidence')
            if confidence is not None and confidence < 0.99:
                msg = random.choice(['啥', '你说啥', '什么']) + random.choice(['我没听清', ''])
                dispatcher.utter_message(msg)
                return []
        for handler in handlers:
            if not handler.match(tracker, domain):
                continue
            events = handler.process(dispatcher, tracker, domain)
            if events is None:
                continue
            msg = '\n'.join([f'{key}: {val}' for key, val in tracker.latest_message.items()])
            logger.debug(f'Handler {handler} processed \u001b[32m{msg}\u001b[0m')
            return events
        dispatcher.utter_template('utter_default', tracker)
        return []

