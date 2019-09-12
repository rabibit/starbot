from .wifi import WifiHandler
from .ask_if_ferry import AskIfFerryHandler
from .bye import ByeHandler
from .simple import SimpleHandler
from .charger import ChargerHandler
from .alarm_clock import AlarmClockHandler
from .order import SimpleOrderHandler
from .ask_for_phone_number import AskForPhoneNumberHandler
from .book_room import BookRoomHandler
from .is_there_xxx import SomethingEatHandler
from .handler import BaseHandler
from .quit_form import QuitHandler
from .repeat_utter import RepeatHandler

handlers = [
    RepeatHandler,
    QuitHandler,
    WifiHandler,
    ByeHandler,
    AskIfFerryHandler,
    ChargerHandler,
    AlarmClockHandler,
    SimpleOrderHandler,
    BookRoomHandler,
    AskForPhoneNumberHandler,
    SomethingEatHandler,
    SimpleHandler,
]
