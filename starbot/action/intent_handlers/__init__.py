
from typing import List
from . import wifi
from . import ask_if_ferry
from . import bye
from . import simple
from .handler import BaseHandler

handlers: List[BaseHandler] = [wifi.WifiHandler(),
                               bye.ByeHandler(),
                               ask_if_ferry.AskIfFerryHandler(),
                               simple.SimpleHandler()]
