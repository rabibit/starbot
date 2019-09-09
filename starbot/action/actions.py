import random
import logging
from rasa_sdk import Action

from typing import Text, Dict, Any, List
from rasa_sdk.executor import CollectingDispatcher, Tracker
from starbot.action.intent_handlers import handlers
from starbot.action.intent_handlers.handler import Context, is_last_message_user

logger = logging.getLogger(__name__)


key_intents = {
    'ask_for_wifi_info',
    'ask_for_wifi_password',
    'ask_for_charger',
    'ask_for_awaking',
    'order_something',
    'ask_for_phone_number',
    'is_there_xxx',
}


class MyDispatcher(object):
    def __init__(self):
        self.messages: [str] = []

    def utter_message(self, text: str):

        self.messages.append("。。" + text)


class ProcessIntentAction(Action):
    def name(self):
        return "action_process_intent"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        what_msg = random.choice(['啥', '你说啥', '什么']) + random.choice(['我没听清', ''])
        my_dispatcher = MyDispatcher()
        if tracker.latest_message:
            if is_last_message_user(tracker):
                msg = '\n'.join([f'{key}: {val}' for key, val in tracker.latest_message.items()])
                logger.debug(f'\u001b[32m{msg}\u001b[0m')
                dispatcher.utter_message(f'/intent is {tracker.latest_message.get("intent")}')
                dispatcher.utter_message(f'/entities {tracker.latest_message.get("entities")}')
            intent = tracker.latest_message.get('intent', {}).get('name')
            confidence = tracker.latest_message.get('intent', {}).get('confidence')
            if intent in key_intents and confidence is not None and confidence < 0.9:
                dispatcher.utter_message(what_msg)
                return []

            context = Context(my_dispatcher, tracker, domain)
            all_handlers = [Handler(context) for Handler in handlers]
            context.handlers = all_handlers
            events = None
            for handler in all_handlers:
                if not handler.match():
                    continue
                events = handler.process()
                if events is None:
                    continue
                logger.debug(f'Handler {handler} processed')
                break

            for handler in all_handlers:
                handler.recover()

            merged_message = "".join(my_dispatcher.messages)
            dispatcher.utter_message(merged_message)

            if events is None and not merged_message:
                dispatcher.utter_message(what_msg)

            return events
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

