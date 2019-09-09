from .handler import BaseHandler
from typing import Text, Dict, Any, List
from rasa_sdk.events import Form


class QuitHandler(BaseHandler):
    def match(self) -> bool:
        if not self.tracker.active_form:
            return False
        if not self.tracker.latest_message:
            return False

        commands = set(self.tracker.latest_message.get('commands') or [])

        if {'exit', 'cancel'} & commands:
            return True
        return False

    def process(self) -> List[Dict[Text, Any]]:
        self.utter_message('好的，还有其它需要吗?')
        return [Form(None)]
