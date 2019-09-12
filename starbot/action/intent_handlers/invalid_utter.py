from locale import atoi

from .handler import BaseHandler
from typing import Text, Dict, Any, List, Optional


class InvalidUtterHandler(BaseHandler):

    def match(self) -> bool:
        return True

    def process(self) -> Optional[List[Dict[Text, Any]]]:

        invalid_utter = self.get_slot('invalid_utter')
        if not invalid_utter:
            invalid_utter = 0
        else:
            invalid_utter = atoi(invalid_utter)
        if invalid_utter > 3:
            invalid_utter = 0
            self.tracker.slots['invalid_utter'] = invalid_utter
            self.utter_message('小智已经尝试多次仍然无法完成您的请求，')
            return None
        invalid_utter += 1
        self.tracker.slots['invalid_utter'] = invalid_utter
        return None
