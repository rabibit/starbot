
from typing import List
from . import wifi
from .handler import BaseHandler

handlers: List[BaseHandler] = [wifi.WifiHandler()]
