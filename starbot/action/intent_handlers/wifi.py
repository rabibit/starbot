
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
        dispatcher.utter_template("utter_wifi_info", tracker)
        from starbot.action.intent_handlers import form_to_handlers
        if tracker.active_form.get('name') in form_to_handlers.keys():
            handler = form_to_handlers[tracker.active_form.get('name')]()
            events = handler.process(dispatcher, tracker, domain)
            if events is None:
                return []
            return events
        return []

