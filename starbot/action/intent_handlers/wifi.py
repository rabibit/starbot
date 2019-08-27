
from .handler import BaseHandler
from typing import Text, Dict, Any, List
from rasa_sdk.executor import CollectingDispatcher, Tracker


class WifiHandler(BaseHandler):
    def match(self, tracker: Tracker, domain: Dict[Text, Any]) -> bool:
        if self.is_last_message_user(tracker) and self.get_last_user_intent(tracker) in (
            'ask_for_wifi_info', 'ask_for_wifi_password'
        ):
            return True
        else:
            return False

    def process(self,
                dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("Wifi名称是俊美的拼音全拼，密码是8个8")
        return []

    def continue_form(self):
        return True
