from starbot.action.intent_handlers.handler import BaseHandler
from starbot.action.db_orm import *


class AlarmQueryHandler(BaseHandler):
    def match(self) -> bool:
        return self.get_last_user_intent() == 'alarm_query'

    def process(self):
        caller = self.get_slot('caller')
        alarms = db_orm_alarm_query(Inform, caller)
        count = 0
        messages = ''
        for alarm in alarms:
            count += 1
            messages += alarm.alarm_clock
            messages += '。。'
        if count == 0:
            self.utter_message(f"您尚未设置任何闹钟")
        elif count == 1:
            self.utter_message(f"您设置一个{messages}的闹钟")
        else:
            self.utter_message(f"您已经设置{count}个闹钟，分别是{messages}")

