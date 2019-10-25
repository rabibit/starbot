
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

    def validate(self, recovering: bool):
        if not recovering:
            self.skip_if_no_update_and_intended()
        if self.charger_type is None:
            if recovering:
                self.utter_message('嗯，你要什么充电器呢?')
            else:
                self.utter_message('什么充电器?')
            return False
        return True

    def commit(self):
        self.utter_message(f'{self.charger_type}的充电器, 是吧, 等下服务员给你送过去哈')
