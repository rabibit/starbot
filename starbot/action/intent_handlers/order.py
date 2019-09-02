
from .handler import BaseFormHandler, BaseForm, get_entity_from_message
from typing import Text


class SimpleOrderHandler(BaseFormHandler):
    class Form(BaseForm):
        __tag__ = 'order'
        thing: str
        count: int
        number: int

    form: Form

    def form_trigger(self, intent: Text):
        return intent == 'order_something'

    def find_thing_in_history(self):
        user_utters = [event for event in self.tracker.events[::-1] if event['event'] == 'user']
        for event in user_utters[1:3]:
            thing = get_entity_from_message(event['parse_data'], 'thing')
            if thing:
                return thing

    def validate(self):
        form = self.form

        if form.thing in {'茶叶', '蚊香'}:
            if form.count is None:
                form.count = 1

        if form.thing is None:
            thing = self.find_thing_in_history()
            if thing:
                self.form.put_entity('thing', thing)
            else:
                self.utter_message("请问您需要什么?")
                return False

        if form.count is None:
            self.utter_message("请问您需要多少{}?".format(form.thing))
            return False

        if form.number is None:
            self.utter_message("请问您的房号是多少?")
            return False
        return True

    def commit(self):
        self.utter_message("好的，您要的{}马上为您送过来".format(self.form.thing))
