
from .handler import BaseHandler
from typing import Text, Dict, Any, List, Optional
from starbot.action.db_orm import *


class SomethingEatHandler(BaseHandler):
    def match(self) -> bool:
        if self.is_last_message_user() and self.get_last_user_intent() in (
            'is_there_xxx',
        ):
            return True
        else:
            return False

    def process(self) -> Optional[List[Dict[Text, Any]]]:
        thing = self.get_entity('thing')
        if not thing:
            self.skip()
            return
        result = db_orm_query(Inform, thing, thing)
        product = False
        products = []
        service = False
        services = []
        position = False
        positions = []
        for rt in result:
            if rt.variety == 'product':
                product = True
                products.append(rt)
            if rt.variety == 'service':
                service = True
                services.append(rt)
            if rt.variety == 'position':
                position = True
                positions.append(rt)
        if product:
            self.utter_message("我们这里有：")
            for food in products:
                self.utter_message("{}，{}元".format(food.name, food.price))
        elif service:
            self.utter_message("我们这里有：")
            for ser in services:
                self.utter_message("{}，{}".format(ser.name, ser.service))
        elif position:
            self.utter_message("附近有：")
            for pos in positions:
                if pos.contact:
                    self.utter_message("{}，{}，联系电话{}".format(pos.name, pos.position, pos.contact))
                else:
                    self.utter_message("{}，{}".format(pos.name, pos.position))
        else:
            self.utter_message("不好意思，我们这里没有")
