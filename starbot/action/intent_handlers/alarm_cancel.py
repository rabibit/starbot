from starbot.action.intent_handlers.handler import BaseHandler
from starbot.action.db_orm import *


class AlarmCancelHandler(BaseHandler):
    def match(self) -> bool:
        return self.get_last_user_intent() == 'alarm_cancel'

    def process(self):
        caller = self.get_slot('caller')
        db_orm_alarm_cancel(Inform, caller)
        self.utter_one_of(
            "闹钟已全部删除",
            "您的闹钟已删除",
        )

