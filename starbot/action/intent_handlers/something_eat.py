
from .handler import BaseHandler
from typing import Text, Dict, Any, List
from starbot.action.db_orm import *


class SomethingEatHandler(BaseHandler):
    def match(self) -> bool:
        if self.is_last_message_user() and self.get_last_user_intent() in (
            'is_there_xxx',
        ):
            return True
        else:
            return False

    def process(self) -> List[Dict[Text, Any]]:
        thing = self.get_entity('thing')
        if not thing:
            return None
        products = db_orm_query(Product, thing)
        self.utter_message("我们这里有：")
        for food in products:
            self.utter_message("{}，单价是{}元".format(food.Name, food.Price))
        return []
