# coding: utf8
from rasa.core_sdk import Action
from rasa.core_sdk.events import AllSlotsReset
from rasa.core_sdk.forms import FormAction
import typing

if typing.TYPE_CHECKING:
    from typing import Text
    from rasa.core_sdk.executor import CollectingDispatcher, Tracker


class RoomForm(FormAction):
    def name(self):
        return "room_form"

    @staticmethod
    def required_slots(tracker):
        return ['room_type', 'checkin_time', 'guest_name', 'guest_phone_number', 'confirmed']

    def submit(self, dispatcher, tracker, domain):
        dispatcher.utter_message("完成")
        return [AllSlotsReset()]

    def slot_mappings(self):
        return {
            'room_type': [
                self.from_entity("room_type"),
                self.from_text(),
            ],
            'guest_name': [
                self.from_entity("guest_name"),
                self.from_text(),
            ],
            'guest_phone_number': [
                self.from_entity("guest_phone_number"),
                self.from_text(),
            ],
            'checkin_time': [
                self.from_entity("checkin_time"),
                self.from_text(),
            ],
            'confirmed': [
                self.from_intent(True, intent='ok'),
                self.from_intent(True, intent='confirm')
            ]
        }

    def request_next_slot(self,
                          dispatcher,  # type: CollectingDispatcher
                          tracker,  # type: Tracker
                          domain  # type: Dict[Text, Any]
                          ):
        events = super(RoomForm, self).request_next_slot(dispatcher, tracker, domain)
        if events is not None and len(events) > 0:
            evt = events[0]
            if evt['event'] == 'slot' and evt['name'] == 'requested_slot' and evt['value'] == 'confirmed':
                dispatcher.messages.pop()
                dispatcher.utter_message("确认一下您的信息, 姓名:[{}] 电话：[{}] 预订一间[{}], [{}]入住".format(
                    tracker.get_slot("guest_name"),
                    tracker.get_slot("guest_phone_number"),
                    tracker.get_slot("room_type"),
                    tracker.get_slot("checkin_time"),
                ))
                dispatcher.utter_message("您看有问题吗？")
        return events


