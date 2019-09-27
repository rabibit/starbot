
from .handler import BaseHandler


class AskIfFerryHandler(BaseHandler):
    def match(self) -> bool:
        if self.is_last_message_user() and self.get_last_user_intent() in (
                'ask_if_ferry'
        ):
            return True
        else:
            return False

    def process(self):
        self.utter_message("您好，我们只提供机场的接送服务")
