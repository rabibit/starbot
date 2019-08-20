
from .handler import BaseHandler
from typing import Text, Dict, Any, List, Optional
from rasa_sdk.executor import CollectingDispatcher, Tracker
from rasa_sdk.events import Form
from rasa_sdk.events import SlotSet


class OrderHandler(BaseHandler):
    def match(self, tracker: Tracker, domain: Dict[Text, Any]) -> bool:
        return True

    def process(self,
                dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]) -> Optional[List[Dict[Text, Any]]]:
        intent = self.get_last_user_intent(tracker)
        if intent in {
            'order_something'
        }:
            thing = self.get_entity(tracker, 'thing')
            count = self.get_entity(tracker, 'count')
            room_number = self.get_entity(tracker, 'number')
            events = []
            if thing is not None:
                events.append(SlotSet('thing', thing))
            if count is not None:
                events.append(SlotSet('count', count))
            if room_number is not None:
                events.append(SlotSet('room_number', room_number))
            if thing is not None and count is not None and room_number is not None:
                dispatcher.utter_message("好的，您要的{}马上为您送过来".format(thing))
                events.append(Form(None))
                return events
            else:
                if thing is None:
                    dispatcher.utter_message("请问您想需要什么？")
                    events.append(Form('order'))
                    return events
                elif count is None:
                    dispatcher.utter_message("请问您想需要多少？")
                    events.append(Form('order'))
                    return events
                elif room_number is None:
                    dispatcher.utter_message("请问您的房号是多少？")
                    events.append(Form('order'))
                    return events
        else:
            if tracker.active_form.get('name') == 'order':
                thing = self.get_entity(tracker, 'thing')
                count = self.get_entity(tracker, 'count')
                room_number = self.get_entity(tracker, 'number')
                sthing = tracker.slots['thing']
                scount = tracker.slots['count']
                sroom_number = tracker.slots['number']
                events = []
                if thing is not None:
                    events.append(SlotSet('thing', thing))
                if count is not None:
                    events.append(SlotSet('count', count))
                if room_number is not None:
                    events.append(SlotSet('room_number', room_number))
                if (thing is not None or sthing is not None) \
                        and (count is not None or scount is not None) \
                        and (room_number is not None or sroom_number is not None):
                    dispatcher.utter_message("好的，您要的{}马上为您送过来".format(thing))
                    events.append(Form(None))
                    return events
                else:
                    if sthing is None and thing is None:
                        dispatcher.utter_message("请问您想需要什么？")
                        return events
                    elif scount is None and count is None:
                        dispatcher.utter_message("请问您要多少呢？")
                        return events
                    elif sroom_number is None and room_number is None:
                        dispatcher.utter_message("请问您的房号是多少？")
                        return events
            else:
                return [Form(None)]
        return None
