import logging
from rasa_sdk import Action

from typing import Text, Dict, Any, List
from rasa_sdk.executor import CollectingDispatcher, Tracker
from starbot.action.intent_handlers import handlers
import random

logger = logging.getLogger(__name__)


class ProcessIntentAction(Action):
    def name(self):
        return "action_process_intent"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        what_msg = random.choice(['啥', '你说啥', '什么']) + random.choice(['我没听清', ''])
        if tracker.latest_message:
            if handlers[0].is_last_message_user(tracker):
                dispatcher.utter_message(f'/intent is {tracker.latest_message.get("intent")}')
                dispatcher.utter_message(f'/entities {tracker.latest_message.get("entities")}')
            confidence = tracker.latest_message.get('intent', {}).get('confidence')
            if confidence is not None and confidence < 0.9:
                dispatcher.utter_message(what_msg)
                return []
        for handler in handlers:
            if not handler.match(tracker, domain):
                continue
            events = handler.process(dispatcher, tracker, domain)
            if events is None:
                continue
            msg = '\n'.join([f'{key}: {val}' for key, val in tracker.latest_message.items()])
            logger.debug(f'Handler {handler} processed \u001b[32m{msg}\u001b[0m')
            return events
        dispatcher.utter_message(what_msg)
        return []


if __name__ == "__main__":
    action = ProcessIntentAction()
    tracker = Tracker('',
                      slots={
                          'checkin_time': None,
                          'confirmed': None,
                          'count': ['两个'],
                          'guest_name': None,
                          'guest_phone_number': None,
                          'number': None,
                          'requested_slot': None,
                          'room_type': None,
                          'thing': ['面包'],
                      },
                      latest_message={
                          'intent': {'name': 'info'},
                          'entities': {}
                      },
                      events=[],
                      paused=False,
                      followup_action=None,
                      active_form={'name': 'order'},
                      latest_action_name=None
                      )
    tracker = Tracker('',
                      slots={
                          'checkin_time': None,
                          'confirmed': None,
                          'count': None,
                          'guest_name': None,
                          'guest_phone_number': None,
                          'number': None,
                          'requested_slot': None,
                          'room_type': None,
                          'thing': None,
                      },
                      latest_message={
                          'intent': {'name': 'info'},
                          'entities': [{'entity': 'thing', 'value': '面包'}]
                      },
                      events=[],
                      paused=False,
                      followup_action=None,
                      active_form={'name': 'order'},
                      latest_action_name=None
                      )

    class Dispatcher:
        def utter_message(self, message):
            print(f'utter {message}')

    actions = action.run(Dispatcher(), tracker, {})
    print(actions)

