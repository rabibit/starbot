
from .handler import BaseHandler
from typing import Text, Dict, Any, List


class AskIfFerryHandler(BaseHandler):
    def match(self) -> bool:
        if self.is_last_message_user() and self.get_last_user_intent() in (
                'ask_if_ferry'
        ):
            return True
        else:
            return False

    def process(self) -> List[Dict[Text, Any]]:
        self.utter_message("您好，我们只提供机场的接送服务")
        return []
