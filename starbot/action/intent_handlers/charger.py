
from .handler import BaseFormHandler, BaseForm
from typing import Text


class ChargerHandler(BaseFormHandler):
    class Form(BaseForm):
        __tag__ = 'charger'
        charger_type: Text
    form: Form

    def form_trigger(self, intent: Text):
        return intent == 'ask_for_charger'

    def validate(self):
        if self.form.charger_type is None:
            self.utter_message('什么充电器?')
            return False
        return True

    def commit(self):
        self.utter_message('好的, 服务员一会儿给你送过去')

