
from .handler import BaseFormHandler, BaseForm
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

    def validate(self):
        form = self.form

        if form.thing in {'茶叶', '蚊香'}:
            if form.count is None:
                form.count = 1

        if form.thing is None:
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
