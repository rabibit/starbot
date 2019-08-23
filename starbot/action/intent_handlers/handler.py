from collections import defaultdict
from typing import Text, Dict, Any, List, Optional

from rasa_sdk.events import SlotSet, Form, AllSlotsReset
from rasa_sdk.executor import CollectingDispatcher, Tracker


class BaseHandler:
    def match(self, tracker: Tracker, domain: Dict[Text, Any]) -> bool:
        """
        :param tracker:
        :param domain:
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

    def process(self,
                dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]) -> Optional[List[Dict[Text, Any]]]:
        """
        :param dispatcher:
        :param tracker:
        :param domain:
        :return: a list of events
        """
        raise NotImplementedError

    @staticmethod
    def is_last_message_user(tracker: Tracker):
        return is_last_message_user(tracker)

    @staticmethod
    def get_last_user_intent(tracker: Tracker):
        return get_user_intent(tracker)

    @staticmethod
    def get_entity(tracker: Tracker, name: Text) -> Optional[Text]:
        if not tracker.latest_message:
            return None
        for entity in tracker.latest_message.get('entities', []):
            if entity['entity'] == name:
                return entity['value']


class BaseForm:
    __tag__ = ''

    def __init__(self, tracker: Tracker):
        if not hasattr(self, '__annotations__'):
            self.__annotations__ = {}
        for k in self.__annotations__:
            if not hasattr(self, k):
                setattr(self, k, None)
        self._entities: Dict[(Text, Any)] = defaultdict(list)
        self._tracker = tracker
        self._fill()

    def _fill(self):
        for k, type_ in self.__annotations__.items():
            if issubclass(type_, List):
                raise NotImplementedError
            entity = self.get_entity(k)
            slot = self.get_slot(k)
            setattr(self, k, slot)
            if entity is not None:
                self._entities[k].append(entity)
        for k, v in self._entities.items():
            # TODO: support multiple values slots
            setattr(self, k, v[0])

    def get_slot(self, name) -> Any:
        return self._tracker.slots.get(name)

    def get_entity(self, name: Text) -> Optional[Text]:
        if not self._tracker.latest_message:
            return None
        for entity in self._tracker.latest_message.get('entities', []):
            if entity['entity'] == name:
                return entity['value']

    def slot_filling_events(self):
        rv = []
        for k, v in self._entities.items():
            rv.append(SlotSet(k, v[0]))
        return rv


class SkipThisHandler(Exception):
    pass


class BaseFormHandler(BaseHandler):
    events: List[Any]
    tracker: Tracker
    dispatcher: CollectingDispatcher
    domain: Dict[Text, Any]

    class Form(BaseForm):
        __tag__ = ''

    form: Form

    @property
    def form_name(self):
        return self.Form.__tag__

    def match(self, tracker: Tracker, domain: Dict[Text, Any]) -> bool:
        return True

    def skip(self):
        raise SkipThisHandler()

    def set_slot(self, name, value):
        self.events.append(SlotSet(name, value))

    def clear_slot(self, name):
        self.set_slot(name, None)

    def process(self,
                dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]) -> Optional[List[Dict[Text, Any]]]:
        self.dispatcher = dispatcher
        self.tracker = tracker
        self.domain = domain
        self.events = []
        try:
            events = self._do_process()
            return self.events + events
        except SkipThisHandler:
            return None

    def _do_process(self) -> List[Dict[Text, Any]]:
        tracker = self.tracker
        trigger = self.form_trigger(get_user_intent(tracker))
        if not (trigger or tracker.active_form and tracker.active_form.get('name') == self.form_name):
            return self.skip()
        # TODO: 有激活表单但是有些意图可能导致切换表单
        self.form = self.Form(tracker)
        if self.validate():
            self.commit()
            return [AllSlotsReset(), Form(None)]
        else:
            events = self.form.slot_filling_events()
            if trigger:
                events.insert(0, Form(self.form_name))
            return events

    def utter_message(self, message):
        self.dispatcher.utter_message(message)

    def form_trigger(self, intent: Text):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def commit(self):
        raise NotImplementedError


def get_user_intent(tracker: Tracker):
    msg = tracker.latest_message
    return msg and msg.get('intent', {}).get('name') or None


def is_last_message_user(tracker: Tracker):
    return tracker.events and tracker.events[-1]['event'] == 'user'
