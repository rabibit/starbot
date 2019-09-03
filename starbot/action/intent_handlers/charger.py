
from .handler import BaseFormHandler, BaseForm
from typing import Text


class ChargerHandler(BaseFormHandler):
    class Form(BaseForm):
        __tag__ = 'charger'
        charger_type: Text
        brand: Text
    form: Form

    def form_trigger(self, intent: Text):
        return intent == 'ask_for_charger'

    @property
    def charger_type(self):
        return self.form.charger_type or self.form.brand

    def validate(self):
        if self.charger_type is None:
            self.utter_message('什么充电器?')
            return False
        return True

    def commit(self):
        self.utter_message(f'{self.charger_type}的充电器, 是吧, 等下服务员给你送过去哈')
