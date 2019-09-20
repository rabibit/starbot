import logging

from functools import reduce
from datetime import date, datetime

from starbot.action.intent_handlers.handler import BaseFormHandler, BaseForm
from typing import Text, Any, Optional

from starbot.nlu.timeparser.timeparser import extract_times, TimePoint
from starbot.action.db_orm import *


logger = logging.getLogger(__name__)


def extract_time(text):
    times = list(extract_times(text or ''))
    logger.info(f'times: {times}')
    if not times:
        return None
    return reduce(lambda x, y: x + y, times)


def hour_to_apm_words(hour):
    if 0 <= hour < 5:
        return '凌晨'
    elif hour < 9:
        return '早上'
    elif hour < 12:
        return '上午'
    elif hour < 13:
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
    return map.get(delta.days+2) or f'{time.month}月{time.day}浩'


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


class AlarmClockHandler(BaseFormHandler):
    class Form(BaseForm):
        __tag__ = 'alarm_clock'
        time: TimePoint
        caller: str

        def get_entity(self, name: Text) -> Optional[Any]:
            if name == 'time':
                t, need_update = self.get_time()
                if need_update:
                    return t
                else:
                    return None
            return super().get_entity(name)

        def slot_decode(self, name, value):
            if name == 'time' and value is not None:
                return TimePoint(value)
            return value

        def slot_encode(self, name, value):
            if name == 'time' and value is not None:
                return value.dump_to_dict()
            return value

        def get_time(self):
            need_update = False
            sentence = self._delegate.tracker.latest_message['text']
            t0 = self.get_slot('time')
            t1 = extract_time(sentence)

            if t1 is not None:
                logger.info('/parsed time: {} -> {}'.format(sentence, t1))
                need_update = True

            if t0 is None:
                return t1, need_update
            else:
                logger.info('/prev time: {}'.format(t0))
                if t1 is not None:
                    logger.info('/new time: {}'.format(t1))
                    t0.update(t1)
                    logger.info('/merged time: {}'.format(t0))
                return t0, need_update

    form: Form

    def form_trigger(self, intent: Text):
        return intent == 'ask_for_awaking'

    def validate(self, recovering):
        if not recovering:
            self.skip_if_no_update_and_intended()
        form = self.form
        if form.time is None:
            self.utter_message("嗯，啥时候提醒您?")
            return False
        elif form.time.hour is None:
            self.utter_message(f"嗯，几点钟提醒您?")
            return False
        else:
            return True

    def commit(self):
        time_words = time_to_human_words(self.form.time)
        caller = self.get_slot('caller')
        alarms = db_orm_alarm_query(Inform, caller, time_words)
        count = 0
        for alarm in alarms:
            count += 1
        if count > 0:
            self.utter_message('闹钟已存在')
        else:
            self.utter_message(f'好的，{time_words}，到时间我会叫你的')
            db_orm_add(Inform(name=caller, variety='alarm_clock', alarm_clock=time_words))

    def match(self) -> bool:
        return True


if __name__ == '__main__':
    import doctest
    doctest.testmod()
