
from .handler import BaseHandler
from typing import Text, Dict, Any, List, Optional
from rasa_sdk.executor import CollectingDispatcher, Tracker
from rasa_sdk.events import Form


class ChargerHandler(BaseHandler):
    def match(self, tracker: Tracker, domain: Dict[Text, Any]) -> bool:
        return True

    def process(self,
                dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]) -> Optional[List[Dict[Text, Any]]]:
        intent = self.get_last_user_intent(tracker)
        if intent in {
            'ask_for_charger'
        }:
            # 问充电器
            cgr_type = self.get_entity(tracker, 'charger_type')
            if cgr_type is None:
                dispatcher.utter_message("你要什么充电器?")
                return [Form('charger')]
            else:
                dispatcher.utter_message("好的, 一会儿服务员给你拿过去")
                return [Form(None)]
        else:
            if tracker.active_form.get('name') == 'charger':
                cgr_type = self.get_entity(tracker, 'charger_type')
                if cgr_type is not None:
                    # 找到charger type后完结表单
                    dispatcher.utter_message('好的')
                    return [Form(None)]
                else:
                    dispatcher.utter_message("什么充电器？")
                    return []
        return None

