import random
import logging
from typing import Text, Dict, Any, List, Optional

from rasa_sdk.events import SlotSet, Form, AllSlotsReset
from rasa_sdk.executor import CollectingDispatcher, Tracker


logger = logging.getLogger(__name__)


class SkipThisHandler(Exception):
    pass


class Abort(Exception):
    pass


TOP_INTENTS = {
    "account_issues",
    "air_conditioner_problem",
    "ask_for_awaking",
    "ask_for_changing_room",
    "ask_for_charger",
    "ask_for_help",
    "ask_for_laundry",
    "ask_for_more_breakfast_ticket",
    "ask_for_phone_number",
    "ask_for_price",
    "ask_for_something_to_eat",
    "ask_for_traffic_info",
    "ask_for_wifi_info",
    "ask_for_wifi_password",
    "ask_price_for_changing_room",
    "ask_to_change_thing",
    "ask_to_clean_room",
    "ask_to_open_door",
    "book_room",
    "breakfast_ticket_not_found",
    "bye",
    "can_i_have_invoice",
    "can_order_meal",
    "cancel_book_room",
    "checkout",
    "confirm_extend_condition",
    "consultation",
    "delay_checkin",
    "greet",
    "how_far?",
    "how_much_did_i_spend",
    "how_much_if_stay_until",
    "how_to_call",
    "is_manager_there",
    "is_my_cloth_ready",
    "is_my_room_ready",
    "is_there_any_massage",
    "is_there_breakfast_now",
    "is_there_cloth_drier",
    "is_there_night_snack",
    "is_there_xxx",
    "is_there_xxx_around",
    "is_vip_the_same",
    "lack_of_thing",
    "laundry_request",
    "leave_over_something",
    "network_problem",
    "order_something",
    "other",
    "other_issue_needs_service",
    "query_agreement_price",
    "query_book_record",
    "query_checkout_time",
    "query_supper_time",
    "room_available",
    "stay_extension",
    "tv_problem",
    "urge",
    "wanna_more",
    "what_can_you_do",
    "when_to_have_breakfast",
    "when_to_have_lunch",
    "where_is_laundry_room",
    "where_is_the_wenxiang",
    "where_is_tv_controller",
    "where_to_have_breakfast",
}
SUB_INTENTS = {
    "any_other?",
    "ask_how_to_pay",
    "ask_if_ferry",
    "buy_or_borrow",
    "can_deliver?",
    "change_info",
    "complain",
    "confirm_location",
    "enha",
    "fetch_it_myself?",
    "hmm",
    "info",
    "is_breakfast_included",
    "is_it_free",
    "is_it_ok",
    "no",
    "not_found",
    "number_of_thing",
    "ok",
    "repeat_confirm",
    "something_like",
    "this_phone",
}


def say_what():
    return random.choice(['啥', '你说啥', '什么']) + random.choice(['我没听清', ''])


class Context:
    handlers: List['BaseHandler']

    def __init__(self,
                 dispatcher: CollectingDispatcher,
                 tracker: Tracker,
                 domain: Dict[Text, Any]):
        self.dispatcher = dispatcher
        self.tracker = tracker
        self.domain = domain
        self.handlers = []

    def cancel_form(self, force=False):
        for handler in self.handlers:
            # TODO: O(n) optimization
            if not handler.is_active():
                continue
            try:
                events = handler.cancel(force)
                logger.info(f"[cancel ] canceled: {handler}")
            except Abort:
                logger.info(f'[cancel ] aborted : {handler}')
                return None
            if events is not None:
                return events
        else:
            return None

    def process(self):
        events = None
        for handler in self.handlers:
            if not handler.match():
                logger.info(f'[process] ignored : {handler}')
                continue
            try:
                events = handler.process()
                if events is None:
                    logger.info(f'[process] pass    : {handler} events: {events}')
                else:
                    logger.info(f'[process] accepted: {handler} events: {events}')
            except Abort:
                logger.info(f'[process] aborted : {handler}')
                events = []
            if events is None:
                continue
            events.append(SlotSet('invalid_utter', 0))
            break
        else:
            invalid_utter = self.tracker.slots.get('invalid_utter')
            if not invalid_utter:
                invalid_utter = 0
            if invalid_utter + 1 > 2:
                # invalid_utter = 0
                # events = [SlotSet('invalid_utter', invalid_utter)]
                self.dispatcher.utter_message('小智已经尝试多次仍然无法完成您的请求，如果您想退出请说退出或者返回')
            else:
                invalid_utter += 1
                events = [SlotSet('invalid_utter', invalid_utter)]
                self.tracker.slots['invalid_utter'] = invalid_utter
        has_words = bool(self.dispatcher.messages)
        # TODO: O(n) optimization
        for handler in self.handlers:
            if not handler.is_active():
                continue
            try:
                handler.recover()
                logger.info(f'[recover] recovered: {handler}')
            except Abort:
                logger.info(f'[recover] aborted  : {handler}')
                break

        messages = []
        if events is None and not has_words:
            messages.append(say_what())
        messages.extend(self.dispatcher.messages)
        merged_message = "。。".join(messages)
        return merged_message, events


class BaseHandler:
    def __init__(self, context: Context):
        self.context = context
        self.dispatcher = context.dispatcher
        self.tracker = context.tracker
        self.domain = context.domain

    def match(self) -> bool:
        """
        :return: True 能处理, False 不能处理

        All intents:
        account_issues                  :0.000
        air_conditioner_problem         :0.000
        any_other?                      :0.000
        ask_for_awaking                 :0.000
        ask_for_changing_room           :0.000
        ask_for_charger                 :0.000
        ask_for_help                    :0.000
        ask_for_laundry                 :0.000
        ask_for_more_breakfast_ticket   :0.000
        ask_for_phone_number            :0.000
        ask_for_price                   :0.000
        ask_for_something_to_eat        :0.000
        ask_for_traffic_info            :0.000
        ask_for_wifi_info               :0.000
        ask_for_wifi_password           :0.000
        ask_how_to_pay                  :0.000
        ask_if_ferry                    :0.000
        ask_price_for_changing_room     :0.000
        ask_to_change_thing             :0.000
        ask_to_clean_room               :0.000
        ask_to_open_door                :0.000
        book_room                       :0.000
        breakfast_ticket_not_found      :0.000
        buy_or_borrow                   :0.000
        bye                             :0.000
        can_deliver?                    :0.000
        can_i_have_invoice              :0.000
        can_order_meal                  :0.000
        cancel_book_room                :0.000
        change_info                     :0.000
        checkout                        :0.000
        complain                        :0.000
        confirm_extend_condition        :0.000
        confirm_location                :0.000
        consultation                    :0.000
        delay_checkin                   :0.000
        enha                            :0.000
        fetch_it_myself?                :0.000
        greet                           :1.000
        hmm                             :0.000
        how_far?                        :0.000
        how_much_did_i_spend            :0.000
        how_much_if_stay_until          :0.000
        how_to_call                     :0.000
        info                            :0.000
        is_breakfast_included           :0.000
        is_it_free                      :0.000
        is_it_ok                        :0.000
        is_manager_there                :0.000
        is_my_cloth_ready               :0.000
        is_my_room_ready                :0.000
        is_there_any_massage            :0.000
        is_there_breakfast_now          :0.000
        is_there_cloth_drier            :0.000
        is_there_night_snack            :0.000
        is_there_xxx                    :0.000
        is_there_xxx_around             :0.000
        is_vip_the_same                 :0.000
        lack_of_thing                   :0.000
        laundry_request                 :0.000
        leave_over_something            :0.000
        network_problem                 :0.000
        no                              :0.000
        not_found                       :0.000
        number_of_thing                 :0.000
        ok                              :0.000
        order_something                 :0.000
        other                           :0.000
        other_issue_needs_service       :0.000
        query_agreement_price           :0.000
        query_book_record               :0.000
        query_checkout_time             :0.000
        query_supper_time               :0.000
        repeat_confirm                  :0.000
        room_available                  :0.000
        something_like                  :0.000
        stay_extension                  :0.000
        this_phone                      :0.000
        tv_problem                      :0.000
        urge                            :0.000
        wanna_more                      :0.000
        what_can_you_do                 :0.000
        when_to_have_breakfast          :0.000
        when_to_have_lunch              :0.000
        where_is_laundry_room           :0.000
        where_is_the_wenxiang           :0.000
        where_is_tv_controller          :0.000
        where_to_have_breakfast         :0.000
        """
        raise NotImplementedError

    def process(self) -> Optional[List[Dict[Text, Any]]]:
        """
        :return: a list of events
        """
        raise NotImplementedError

    def is_last_message_user(self):
        return is_last_message_user(self.tracker)

    def get_last_user_intent(self):
        return get_user_intent(self.tracker)

    def is_top_intent(self):
        return self.get_last_user_intent() in TOP_INTENTS

    def get_entity(self, name: Text) -> Optional[Text]:
        if not self.tracker.latest_message:
            return None
        return get_entity_from_message(self.tracker.latest_message, name)

    def get_slot(self, name: Text) -> Any:
        return self.tracker.slots.get(name)

    def utter_message(self, message):
        self.dispatcher.utter_message(message)

    def recover(self):
        pass

    @staticmethod
    def cancel(force: bool):
        return None

    def is_active(self):
        return False

    def abort(self):
        # self.processed = False
        raise Abort()

    def __repr__(self):
        return type(self).__name__


class BaseForm:
    """
    >>> class MyForm(BaseForm):
    ...  a: str = "a"
    ...  b: str
    ...
    >>> f = MyForm(None)
    >>> f.a
    'a'
    >>> f.b = 'b'
    >>> f.b
    'b'
    >>> f.slot_filling_events()
    [{'event': 'slot', 'timestamp': None, 'name': 'b', 'value': 'b'}]
    >>> f.a = None
    >>> sorted(f.slot_filling_events(), key=lambda x: x['name'])
    [{'event': 'slot', 'timestamp': None, 'name': 'a', 'value': None}, {'event': 'slot', 'timestamp': None, 'name': 'b', 'value': 'b'}]
    """

    __tag__ = ''

    def __init__(self, *, delegate=None, from_slot_only=False):
        self._delegate = delegate
        if not hasattr(self, '__annotations__'):
            self.__annotations__ = {}
        self.__attrs__ = {}
        self.__dirty_attrs__ = set()

        for k in self.__annotations__:
            self.__attrs__[k] = getattr(type(self), k, None)

        if delegate is not None:
            self._fill(from_slot_only)

    def __getattribute__(self, item):
        if item == '__annotations__' or item not in self.__annotations__:
            return super(BaseForm, self).__getattribute__(item)
        else:
            return self.__attrs__[item]

    def __setattr__(self, item, value):
        if item in self.__annotations__:
            if (value, self.__attrs__[item]) != (None, None):
                self.__dirty_attrs__.add(item)
                self.__attrs__[item] = value
        else:
            super(BaseForm, self).__setattr__(item, value)

    def _fill(self, from_slot_only):
        for k, type_ in self.__annotations__.items():
            entity = self.get_entity(k) if not from_slot_only else None
            slot = self.get_slot(k)
            self.__attrs__[k] = slot
            if entity is not None:
                setattr(self, k, entity)

    def get_slot(self, name) -> Any:
        return self.slot_decode(name, self._delegate.get_slot(name))

    def get_entity(self, name: Text) -> Optional[Text]:
        return self._delegate.get_entity(name)

    def slot_filling_events(self):
        rv = []
        for k in self.__dirty_attrs__:
            v = self.slot_encode(k, self.__attrs__[k])
            rv.append(SlotSet(k, v))
        return rv

    def is_updated(self):
        return len(self.__dirty_attrs__) > 0

    def slot_encode(self, name, value):
        return value

    def slot_decode(self, name, value):
        return value


class BaseFormHandler(BaseHandler):
    events: List[Any]
    tracker: Tracker
    dispatcher: CollectingDispatcher
    domain: Dict[Text, Any]

    class Form(BaseForm):
        __tag__ = ''

    form: Form
    processed = False

    @property
    def form_name(self):
        return self.Form.__tag__

    def match(self) -> bool:
        return True

    def skip(self):
        self.processed = False
        raise SkipThisHandler()

    def skip_if_no_update_and_intended(self):
        if not self.form.is_updated():
            self.skip_if_intended()

    def skip_if_intended(self):
        if not self.form_trigger(self.get_last_user_intent()):
            self.skip()

    def set_slot(self, name, value):
        self.events.append(SlotSet(name, value))

    def get_slot(self, name):
        return self.tracker.slots.get(name)

    def clear_slot(self, name):
        self.set_slot(name, None)

    def process(self) -> Optional[List[Dict[Text, Any]]]:
        self.events = []
        try:
            return self._do_process()
        except SkipThisHandler:
            return None

    def is_active(self):
        return self.tracker.active_form and self.tracker.active_form.get('name') == self.form_name

    def _do_process(self) -> List[Dict[Text, Any]]:
        self.processed = True
        tracker = self.tracker
        trigger = self.form_trigger(get_user_intent(tracker))
        if not (trigger or self.is_active()):
            return self.skip()
        # TODO: 有激活表单但是有些意图可能导致切换表单
        self.form = self.Form(delegate=self)
        if self.validate(recovering=False):
            self.commit()
            events = self.events + [AllSlotsReset(), Form(None)]
        else:
            events = self.form.slot_filling_events() + self.events
            if trigger:
                events.insert(0, Form(self.form_name))
                if not self.is_active():
                    events.insert(0, AllSlotsReset())
        return events

    def recover(self):
        if self.processed:
            return
        if self.is_active():
            self.form = self.Form(delegate=self, from_slot_only=True)
            self.validate(recovering=True)

    def cancel(self, force: bool):
        self.processed = True
        return [Form(None), AllSlotsReset()] if self.is_active() else None

    def form_trigger(self, intent: Text):
        raise NotImplementedError

    def validate(self, recovering: bool):
        raise NotImplementedError

    def commit(self):
        raise NotImplementedError


def get_user_intent(tracker: Tracker):
    msg = tracker.latest_message
    return msg and msg.get('intent', {}).get('name') or None


def is_last_message_user(tracker: Tracker):
    return tracker.events and tracker.events[-1]['event'] == 'user'


def get_entity_from_message(message: Dict[Text, Any], name: Text):
    for entity in message.get('entities', []):
        if entity['entity'] == name:
            return entity['value']


def get_entities_from_message(message: Dict[Text, Any], name: Text):
    all_entities = []
    for entity in message.get('entities', []):
        if entity['entity'] == name:
            all_entities.append(entity['value'])
    return all_entities
