
from .handler import BaseHandler
from typing import Text, Dict, Any, List, Optional
from rasa_sdk.executor import CollectingDispatcher, Tracker
from rasa_sdk.events import Form


class AskForPhoneNumberHandler(BaseHandler):
    def match(self, tracker: Tracker, domain: Dict[Text, Any]) -> bool:
        return True

    def process(self,
                dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]) -> Optional[List[Dict[Text, Any]]]:
        intent = self.get_last_user_intent(tracker)
        if intent in {
            'ask_for_phone_number'
        }:
            subject = self.get_entity(tracker, 'subject_of_phone_number')
            if subject is None:
                dispatcher.utter_message("请问您要查询什么电话号码?")
                return [Form('subject_of_phone_number')]
            else:
                dispatcher.utter_message('{}的电话号码是123456789'.format(subject))
                return [Form(None)]
        else:
            if tracker.active_form.get('name') == 'subject_of_phone_number':
                subject = self.get_entity(tracker, 'subject_of_phone_number')
                if subject is not None:
                    # 找到subject_of_phone_number后完结表单
                    dispatcher.utter_message('{}的电话号码是123456789'.format(subject))
                    return [Form(None)]
                else:
                    dispatcher.utter_message("请问您要查询什么电话号码?")
                    return []
        return None

    def continue_form(self):
        return False
