
from .handler import BaseHandler
from typing import Text, Dict, Any, List


class ByeHandler(BaseHandler):
    def match(self) -> bool:
        if self.is_last_message_user() and self.get_last_user_intent() in (
            'bye'
        ):
            return True
        else:
            return False

    def process(self):
        self.utter_message("再见")
