import logging

from datetime import date, datetime
from starbot.action.intent_handlers.handler import BaseHandler
from typing import Text, Dict, Any, List, Optional
from rasa_sdk.executor import CollectingDispatcher, Tracker
from rasa_sdk.events import Form, SlotSet

from starbot.nlu.timeparser.timeparser import extract_times, TimePoint


logger = logging.getLogger(__name__)


def extract_time(text):
    times = list(extract_times(text or ''))
    if times:
        return times[-1]
    else:
        return None


def hour_to_apm_words(hour):
    if 0 <= hour < 5:
        return '凌晨'
    elif hour < 9:
        return '早上'
    elif hour < 12:
        return '上午'
    elif hour < 1:
        return '中午'
    elif hour < 18:
        return '下午'
    else:
        return '晚上'


def apm_to_words(ampm):
    return {
        'midnight': '凌晨',
        'morning': '上午',
        'noon': '中午',
        'afternoon': '下午',
        'night': '晚上',
    }.get(ampm, '')


def time_to_day_words(time: datetime):
    today = date.today()
    delta = time.date() - today
    map = dict(enumerate(['前天', '昨天', '今天', '明天', '后天']))
    return map.get(delta.days+2) or f'{time.month}月{time.day}号'


def time_to_human_words(tp: TimePoint):
    """

    :param tp:
    :return:

    >>> time_to_human_words(TimePoint('明天'))
    '明天'
    >>> time_to_human_words(TimePoint('明天上午'))
    '明天上午'
    >>> time_to_human_words(TimePoint('明天早上9点'))
    '明天上午9点钟'
    >>> time_to_human_words(TimePoint('明天早上9点10'))
    '明天上午9点10分'

    """
    time = tp.get_datetime(True)
    logger.info(time.strftime('time_to_human_words: time is %Y-%m-%d %H:%M:%S'))
    words = time_to_day_words(time)
    if tp.hour is not None:
        hour = time.hour
        words += hour_to_apm_words(hour)
        if hour > 12:
            hour -= 12
        words += f'{hour}点'
        if time.minute is None or time.minute == 0:
            words += '钟'
        else:
            words += f'{time.minute}分'
    elif tp.ampm:
        words += apm_to_words(tp.ampm)
    return words


class AlarmClockHandler(BaseHandler):
    def match(self, tracker: Tracker, domain: Dict[Text, Any]) -> bool:
        return True

    def get_time(self, tracker: Tracker):
        need_update = False
        alarm_time = self.get_entity(tracker, 'time')
        t0 = self.get_slot(tracker, 'time')
        t1 = extract_time(alarm_time)

        if t1 is not None:
            logger.info('/parsed {} -> {}'.format(alarm_time, t1))
            need_update = True

        if t0 is None:
            return t1, need_update
        else:
            t0 = TimePoint(t0)
            if t1 is not None:
                t0.update(t1)
            return t0, need_update

    def process(self,
                dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]) -> Optional[List[Dict[Text, Any]]]:
        intent = self.get_last_user_intent(tracker)
        if intent in {
            'ask_for_awaking'
        }:
            time, _ = self.get_time(tracker)

            if time is None:
                dispatcher.utter_message("好的，啥时候提醒您?")
                return [Form('alarm_clock')]
            else:
                if time.hour is None:
                    dispatcher.utter_message(f"好的，几点钟提醒您?")
                    return [Form('alarm_clock'), SlotSet('time', time.dump_to_dict())]
                else:
                    time_words = time_to_human_words(time)
                    dispatcher.utter_message(f'{time_words}好的，到时间我会电话给你')
                    return [Form(None)]
        else:
            if tracker.active_form.get('name') == 'alarm_clock':
                time, _ = self.get_time(tracker)
                if time and time.hour is not None:
                    time_words = time_to_human_words(time)
                    dispatcher.utter_message(f'好的，{time_words}我会电话给你')
                    return [Form(None)]
                else:
                    dispatcher.utter_message("啥时候提醒你?")
                    return []
        return None


if __name__ == '__main__':
    import doctest
    doctest.testmod()
