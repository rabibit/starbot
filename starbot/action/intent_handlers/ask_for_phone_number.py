
from .handler import BaseFormHandler, BaseForm
from typing import Text


class AskForPhoneNumberHandler(BaseFormHandler):
    class Form(BaseForm):
        __tag__ = 'subject_of_phone_number'
        subject_of_phone_number: str
        person: str

    def form_trigger(self, intent: Text):
        return intent == 'ask_for_phone_number'

    def validate(self, recovering: bool):
        if self.form.subject_of_phone_number is None and self.form.person is None:
            self.utter_message("请问您要查询什么电话号码?")
            return False
        return True

    def commit(self):
        self.utter_message(
            f'{self.form.subject_of_phone_number if self.form.subject_of_phone_number is not None else self.form.person}的电话号码是123456789')
