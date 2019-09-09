from .handler import BaseHandler
from typing import Text, Dict, Any, List


class RepeatHandler(BaseHandler):
    def match(self) -> bool:
        commands = set(self.tracker.latest_message.get('commands') or [])
        return 'what?' in commands

    def process(self) -> List[Dict[Text, Any]]:
        latest_bot_message = None

        for event in self.tracker.events[::-1]:
            if event['event'] == 'bot' and not event['text'].startswith('/'):
                latest_bot_message = event['text']
                break

        if not latest_bot_message:
            self.utter_message('我没说啥呀')
        else:
            if not latest_bot_message.startswith('我说:'):
                latest_bot_message = "我说:" + latest_bot_message
            self.utter_message(latest_bot_message)

        return []
