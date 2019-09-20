
from .handler import BaseHandler
from typing import Text, Dict, Any, List


class WifiHandler(BaseHandler):
    def match(self) -> bool:
        if self.is_last_message_user() and self.get_last_user_intent() in (
            'ask_for_wifi_info', 'ask_for_wifi_password'
        ):
            return True
        else:
            return False

    def process(self):
        self.utter_message("Wifi名称是俊美的拼音全拼，密码是8个8")
