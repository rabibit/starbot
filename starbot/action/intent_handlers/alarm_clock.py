
from .handler import BaseHandler
from typing import Text, Dict, Any, List, Optional
from rasa_sdk.executor import CollectingDispatcher, Tracker
from rasa_sdk.events import Form


class AlarmClockHandler(BaseHandler):
    def match(self, tracker: Tracker, domain: Dict[Text, Any]) -> bool:
        return True

    def process(self,
                dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]) -> Optional[List[Dict[Text, Any]]]:
        intent = self.get_last_user_intent(tracker)
        if intent in {
            'ask_for_awaking'
        }:
            # 问闹钟设置时间
            alarm_time = self.get_entity(tracker, 'time')
            if alarm_time is None:
                dispatcher.utter_message("请问您要设置几点的闹钟?")
                return [Form('alarm_clock')]
            else:
                dispatcher.utter_message('好的，闹钟已设置，时间为{}'.format(alarm_time))
                return [Form(None)]
        else:
            if tracker.active_form.get('name') == 'alarm_clock':
                alarm_time = self.get_entity(tracker, 'time')
                if alarm_time is not None:
                    # 找到alarm_time后完结表单
                    dispatcher.utter_message('好的，闹钟已设置，时间为{}'.format(alarm_time))
                    return [Form(None)]
                else:
                    dispatcher.utter_message("请问您要设置几点的闹钟?")
                    return []
        return None

    def continue_form(self):
        return False

