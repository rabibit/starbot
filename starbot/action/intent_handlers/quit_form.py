import re
import logging

from .handler import BaseHandler
from typing import Text, Dict, Any, List, Optional


logger = logging.getLogger(__name__)


class QuitHandler(BaseHandler):
    not_pat = re.compile(r'[没谁]说?[叫让]你|我不要|我没有|我没说|神经病')

    def match(self) -> bool:
        return True

    def process(self) -> Optional[List[Dict[Text, Any]]]:
        if not self.tracker.active_form:
            return None
        if not self.tracker.latest_message:
            return None

        commands = set(self.tracker.latest_message.get('commands') or [])

        events = None
        if {'exit', 'cancel'} & commands:
            events = self.context.cancel_form(force=True)
        else:
            message = self.tracker.latest_message['text']
            logger.info(f'message={message}, match={self.not_pat.search(message)}')
            if self.not_pat.search(message):
                events = self.context.cancel_form(force=False)
                self.abort()
        if events is not None:
            self.utter_message('好的， 还有其它需要吗')
        return events
