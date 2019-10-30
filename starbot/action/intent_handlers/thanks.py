from .handler import BaseHandler


class ThanksHandler(BaseHandler):
    def match(self) -> bool:
        message = self.tracker.latest_message['text']
        return '谢谢' in message

    def process(self):
        self.utter_message("不客气")
