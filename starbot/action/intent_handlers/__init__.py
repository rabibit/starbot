
from typing import List
from .wifi import WifiHandler
from .ask_if_ferry import AskIfFerryHandler
from .bye import ByeHandler
from .simple import SimpleHandler
from .charger import ChargerHandler
from .handler import BaseHandler

handlers: List[BaseHandler] = [WifiHandler(),
                               ByeHandler(),
                               AskIfFerryHandler(),
                               ChargerHandler(),
                               SimpleHandler()
                               ]
