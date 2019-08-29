
from .handler import BaseHandler
from typing import Text, Dict, Any, List, Optional
from rasa_sdk.executor import CollectingDispatcher, Tracker
from rasa_sdk.events import Form

from starbot.nlu.timeparser.timeparser import extract_times


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
            times = list(extract_times(alarm_time or ''))

            if alarm_time:
                dispatcher.utter_message('/parsed {} -> {}'.format(alarm_time, times))

            if not times:
                dispatcher.utter_message("好的，啥时候提醒您?")
                return [Form('alarm_clock')]
            elif times[-1].hour is None:
                dispatcher.utter_message(f"好的，{alarm_time}几点钟提醒您?")
                return [Form('alarm_clock')]
            else:
                time = times[0]
                dispatcher.utter_message('好的，{}我会电话给你'.format(alarm_time))
                return [Form(None)]
        else:
            if tracker.active_form.get('name') == 'alarm_clock':
                alarm_time = self.get_entity(tracker, 'time')
                times = list(extract_times(alarm_time or ''))
                if alarm_time:
                    dispatcher.utter_message('/parsed {} -> {}'.format(alarm_time, times))
                if times and times[-1].hour is not None:
                    # 找到alarm_time后完结表单
                    dispatcher.utter_message('{}好的，到时候我会电话给你'.format(alarm_time))
                    return [Form(None)]
                else:
                    dispatcher.utter_message("啥时候提醒你?")
                    return []
        return None

