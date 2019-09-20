import re
import logging

from .handler import BaseHandler
from typing import Text, Dict, Any, List, Optional


logger = logging.getLogger(__name__)


class QuitHandler(BaseHandler):
    not_pat = re.compile(r'[没谁]说?[叫让]你|我不要|我没有|我没说|神经病')

    def match(self) -> bool:
        return True

    def process(self):
        if not self.tracker.active_form:
            self.skip()
            return
        if not self.tracker.latest_message:
            self.skip()
            return

        commands = set(self.tracker.latest_message.get('commands') or [])

        if {'exit', 'cancel'} & commands:
            canceled = self.context.cancel_form(force=True)
            if canceled:
                self.utter_message('好的， 还有其它需要吗')
                self.abort()
        else:
            message = self.tracker.latest_message['text']
            logger.info(f'message={message}, match={self.not_pat.search(message)}')
            if self.not_pat.search(message):
                self.context.cancel_form(force=False)
                self.utter_message('好的， 还有其它需要吗')
                self.abort()
        self.skip()
