import random
import logging
from rasa_sdk import Action

from typing import Text, Dict, Any, List
from rasa_sdk.executor import CollectingDispatcher, Tracker
from starbot.action.intent_handlers import handlers
from starbot.action.intent_handlers.handler import Context, is_last_message_user, say_what

logger = logging.getLogger(__name__)


key_intents = {
    'greet',
    'bye',
    'ok',
    'consultation',
    'ask_for_price',
    'room_available',
    'enha',
    'what_can_you_do',
    'ask_for_help',
    'ask_for_wifi_password',
    'ask_for_wifi_info',
    'where_is_the_wenxiang',
    'order_something',
    'not_found',
    'is_there_xxx',
    'ask_for_awaking',
    'breakfast_ticket_not_found',
    'where_to_have_breakfast',
    'when_to_have_breakfast',
    'when_to_have_lunch',
    'is_there_breakfast_now',
    'ask_price_for_changing_room',
    'leave_over_something',
    'is_it_free',
    'complain',
    'ask_to_clean_room',
    'network_problem',
    'ask_to_open_door',
    'ask_to_change_thing',
    'ask_for_phone_number',
    'ask_for_traffic_info',
    'delay_checkin',
    'is_my_room_ready',
    'is_my_cloth_ready',
    'ask_for_laundry',
    'cloth_not_dry',
    'how_far?',
    'confirm_location',
    'cancel_book_room',
    'charger_type',
    'can_deliver?',
    'any_other?',
    'is_vip_the_same',
    'repeat_confirm',
    'buy_or_borrow',
    'this_phone',
    'how_can_i_do',
    'query_book_record',
    'ask_if_ferry',
    'how_much_if_stay_until',
    'is_it_ok',
    'query_agreement_price',
    'fetch_it_myself?',
    'can_i_have_invoice',
    'wanna_more',
    'ask_for_more_breakfast_ticket',
    'query_foods',
    'and_xxx',
    'lack_of_thing',
    'how_much_did_i_spend',
    'is_manager_there',
    'meituan_ticket_comfirm',
    'is_breakfast_custom?',
    'i_have_booked_some_room',
    'ask_how_to_pay',
    'ask_for_charger',
    'stay_extension',
    'tv_problem',
    'laundry_request',
    'where_is_laundry_room',
    'other_issue_needs_service',
    'air_conditioner_problem',
    'query_checkout_time',
    'urge',
    'account_issues',
    'how_to_call',
    'confirm_extend_condition',
    'ask_for_changing_room',
    'where_is_tv_controller',
    'is_breakfast_included',
    'query_supper_time',
    'can_order_meal',
    'alarm_cancel',
    'alarm_query',
}


class ProcessIntentAction(Action):
    def name(self):
        return "action_process_intent"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        if tracker.latest_message:
            if is_last_message_user(tracker):
                msg = '\n'.join([f'{key}: {val}' for key, val in tracker.latest_message.items()])
                logger.debug(f'\u001b[32m{msg}\u001b[0m')
                dispatcher.utter_message(f'/intent is {tracker.latest_message.get("intent")}')
                dispatcher.utter_message(f'/entities {tracker.latest_message.get("entities")}')
            else:
                logger.debug(f'last event is {tracker.events[-1]}')
            intent = tracker.latest_message.get('intent', {}).get('name')
            confidence = tracker.latest_message.get('intent', {}).get('confidence')
            if intent in key_intents and confidence is not None and confidence < 0.8:
                dispatcher.utter_message(say_what())
                return []

            context = Context(dispatcher, tracker, domain)
            all_handlers = [Handler(context) for Handler in handlers]
            # context.handlers = sorted(all_handlers, key=lambda x: not x.is_active())
            context.handlers = all_handlers
            return context.process()
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

