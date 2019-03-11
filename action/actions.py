# coding: utf8
from rasa_core_sdk import Action
from rasa_core_sdk.events import AllSlotsReset
from rasa_core_sdk.forms import FormAction
import typing

if typing.TYPE_CHECKING:
    from typing import Text
    from rasa_core_sdk.executor import CollectingDispatcher, Tracker


class BookRoomAskConfirm(Action):
    def name(self):
        # type: () -> Text
        return "ask_confirm"

    def run(self, dispatcher, tracker, domain):
        # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict[Text, Any]]

        return []


class BookRoom(FormAction):
    def name(self):
        return "book_room"

    @staticmethod
    def required_slots(tracker):
        return ['room_type', 'guest_name', 'guest_phone_number', 'checkin_time', 'confirmed']

    def submit(self, dispatcher, tracker, domain):
        dispatcher.utter_message("完成")
        return [AllSlotsReset()]

    def slot_mappings(self):
        return {'confirmed': [
            self.from_intent(True, intent='ok'),
            self.from_intent(True, intent='confirm')
        ]}

    def request_next_slot(self,
                          dispatcher,  # type: CollectingDispatcher
                          tracker,  # type: Tracker
                          domain  # type: Dict[Text, Any]
                          ):
        events = super(BookRoom, self).request_next_slot(dispatcher, tracker, domain)
        if events is not None and len(events) > 0:
            evt = events[0]
            if evt['event'] == 'slot' and evt['name'] == 'requested_slot' and evt['value'] == 'confirmed':
                dispatcher.messages.pop()
                dispatcher.utter_message("确认一下您的信息, 姓名:{} 电话：{} 预订一间{}, {}入住".format(
                    tracker.get_slot("guest_name"),
                    tracker.get_slot("guest_phone_number"),
                    tracker.get_slot("room_type"),
                    tracker.get_slot("checkin_time"),
                ))
        return events


