
from .handler import BaseHandler
from typing import Text, Dict, Any, List
from rasa_sdk.executor import CollectingDispatcher, Tracker
from starbot.action.db_orm import *


class SomethingEatHandler(BaseHandler):
    def match(self, tracker: Tracker, domain: Dict[Text, Any]) -> bool:
        if self.is_last_message_user(tracker) and self.get_last_user_intent(tracker) in (
            'is_there_xxx',
        ):
            return True
        else:
            return False

    def process(self,
                dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        thing = self.get_entity(tracker, 'thing')
        if not thing:
            return None
        products = db_orm_query(Product, thing)
        dispatcher.utter_message("我们这里有：")
        for food in products:
            dispatcher.utter_message("{}，单价是{}元".format(food.Name, food.Price))
        return []

    def continue_form(self):
        return True
