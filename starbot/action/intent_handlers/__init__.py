
from typing import List
from . import wifi
from .handler import HandlerBase

handlers: List[HandlerBase] = [wifi.WifiHandler()]
