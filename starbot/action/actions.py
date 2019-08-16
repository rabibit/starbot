from rasa_sdk import Action

from typing import Text, Dict, Any
from rasa_sdk.executor import CollectingDispatcher, Tracker
from .intent_handlers import handlers


class ProcessIntentAction(Action):
    def name(self):
        return "action_process_intent"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]):
        for handler in handlers:
            if handler.match(tracker, domain):
                if handler.process(dispatcher, tracker, domain):
                    break

