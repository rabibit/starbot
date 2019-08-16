
from .handler import HandlerBase
from typing import Text, Dict, Any
from rasa_sdk.executor import CollectingDispatcher, Tracker


class WifiHandler(HandlerBase):
    def match(self, tracker: Tracker, domain: Dict[Text, Any]) -> bool:
        if self.is_last_message_user(tracker) and self.get_last_user_intent(tracker) in (
            'ask_for_wifi_info', 'ask_for_wifi_password'
        ):
            return True
        else:
            return False

    def process(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]):
        dispatcher.utter_message("Wifi Info")
        pass

