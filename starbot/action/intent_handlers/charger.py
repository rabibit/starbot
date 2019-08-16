
from .handler import BaseHandler
from typing import Text, Dict, Any, List
from rasa_sdk.executor import CollectingDispatcher, Tracker
from rasa_sdk.events import Form


class ChargerHandler(BaseHandler):
    def match(self, tracker: Tracker, domain: Dict[Text, Any]) -> bool:
        intent = self.get_last_user_intent(tracker)
        if intent in {
            'ask_for_charger'
        }:
            return True
        return False

    def process(self,
                dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("What type of charger?")
        return [Form('charger')]

