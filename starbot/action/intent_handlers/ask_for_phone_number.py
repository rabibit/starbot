
from .handler import BaseHandler
from typing import Text, Dict, Any, List, Optional
from rasa_sdk.events import Form


class AskForPhoneNumberHandler(BaseHandler):
    def match(self) -> bool:
        return True

    def process(self) -> Optional[List[Dict[Text, Any]]]:
        intent = self.get_last_user_intent()
        if intent in {
            'ask_for_phone_number'
        }:
            subject = self.get_entity('subject_of_phone_number')
            if subject is None:
                self.utter_message("请问您要查询什么电话号码?")
                return [Form('subject_of_phone_number')]
            else:
                self.utter_message('{}的电话号码是123456789'.format(subject))
                return [Form(None)]
        else:
            if self.tracker.active_form.get('name') == 'subject_of_phone_number':
                subject = self.get_entity('subject_of_phone_number')
                if subject is not None:
                    # 找到subject_of_phone_number后完结表单
                    self.utter_message('{}的电话号码是123456789'.format(subject))
                    return [Form(None)]
                else:
                    self.utter_message("请问您要查询什么电话号码?")
                    return []
        return None
