from .wifi import WifiHandler
from .ask_if_ferry import AskIfFerryHandler
from .bye import ByeHandler
from .simple import SimpleHandler
from .charger import ChargerHandler
from .alarm_clock import AlarmClockHandler
from .order import SimpleOrderHandler
from .ask_for_phone_number import AskForPhoneNumberHandler
from .book_room import BookRoomHandler
from .handler import BaseHandler


intent_to_handlers = {
    'ask_for_wifi_info': WifiHandler,
    'ask_for_wifi_password': WifiHandler,
    'bye': ByeHandler,
    'ask_if_ferry': AskIfFerryHandler,
    'ask_for_charger': ChargerHandler,
    'ask_for_awaking': AlarmClockHandler,
    'order_something': SimpleOrderHandler,
    'book_room': BookRoomHandler,
    'ask_for_phone_number': AskForPhoneNumberHandler,
    'simple': SimpleHandler,
}


form_to_handlers = {
    'charger': ChargerHandler,
    'alarm_clock': AlarmClockHandler,
    'order': SimpleOrderHandler,
    'book_room': BookRoomHandler,
    'subject_of_phone_number': AskForPhoneNumberHandler,
}

