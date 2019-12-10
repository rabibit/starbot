
from .handler import BaseHandler
from typing import Text, Dict, Any, List, Optional


class AgreementPriceHandler(BaseHandler):
    def match(self) -> bool:
        if self.is_last_message_user() and self.get_last_user_intent() in (
            'query_agreement_price',
        ):
            return True
        else:
            return False

    def process(self) -> Optional[List[Dict[Text, Any]]]:
        org = self.get_entity('org')
        if not org:
            self.skip()
            return
        if org in ["星网锐捷", "星网智慧", "智慧科技", "锐捷", "星网", "智慧"]:
            self.utter_message(f"{org}的协议价是: 标间200, 商务套房300")
        else:
            self.utter_message(f"不好意思，{org}暂时没有协议价，不过我们目前有九五折优惠活动，您可以关注一下")
