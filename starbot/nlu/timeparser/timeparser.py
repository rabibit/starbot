import re
import logging
import calendar
from datetime import datetime, timedelta, date, timezone

from typing import Text, Optional, NoReturn, Union, Any, Dict
from starbot.nlu.timeparser.numberify import numberify, WEEK_PREFIX
from starbot.nlu.timeparser.pattern import strict_patterns as patterns


logger = logging.getLogger(__name__)


unreachable = False


class ParseTimeError(Exception):
    pass


def abort(s: str) -> NoReturn:
    raise ParseTimeError(s)


def preprocess(text):
    p = re.compile(r'\s+|[的]+')
    text = p.sub('', text)
    text = numberify(text)
    return text


def get_time_expressions(text):
    text = preprocess(text)
    cur = -1
    tokens = []
    for m in patterns.finditer(text):
        s, e = m.span()
        token = m.group()
        if s == cur:
            tokens[-1] += token
        else:
            tokens.append(token)
        cur = e
    return [TimeExpression(t) for t in tokens]


def extract_times(text):
    """

    :param text:
    :return:

    >>> t = list(extract_times("五分钟之后"))[0]
    >>> dt = t.get_datetime() - datetime.now()
    >>> assert abs(dt.total_seconds() - 300) < 1

    >>> t = list(extract_times("半个小时之后"))[0]
    >>> dt = t.get_datetime() - datetime.now()
    >>> assert abs(dt.total_seconds() - 1800) < 1

    """
    for token in get_time_expressions(text):
        try:
            yield TimePoint(token)
        except ParseTimeError as e:
            logger.warning(f'Parse time token {token.expr} failed: {e}')


def parse_number_with_regex(text, pattern, range=None):
    m = pattern.search(text)
    if m:
        rv = int(m.group())
        if range is not None:
            if rv < range[0] or rv > range[1]:
                return None
        return rv
    return None


def parse_year(text):
    """

    :param text:
    :return:

    >>> parse_year("19年三月")
    2019

    >>> parse_year("三月")

    >>> parse_year("89年三月")
    1989

    >>> parse_year("1889年三月")
    1889

    >>> parse_year("889年三月")
    889
    """

    # 3-4位
    n = parse_number_with_regex(
        text,
        re.compile("[0-9]?[0-9]{3}(?=年)")
    )
    if n is not None:
        return n

    # 2位
    pat = re.compile("[0-9]{2}(?=年)")
    m = pat.search(text)
    if m:
        year = int(m.group())
        if 0 <= year <= 100:
            if year < 30:
                year += 2000
            else:
                year += 1900
        return year
    return None


def parse_month(text):
    """

    :param text:
    :return:

    >>> parse_month("3月")
    3
    >>> parse_month("12月")
    12
    >>> parse_month("13月")
    3
    """
    return parse_number_with_regex(
        text,
        re.compile('((10)|(11)|(12)|([1-9]))(?=月)')
    )


def parse_day(text):
    """

    :param text:
    :return:

    >>> parse_day("30号")
    30

    >>> parse_day("20日")
    20

    >>> parse_day("200日")

    >>> parse_day("月20")
    20

    >>> parse_day("月200")

    """

    return parse_number_with_regex(
        text,
        re.compile(r'((?<!\d))([0-3][0-9]|[1-9])(?=([日号]))|(?<=月)([0-3][0-9]|[1-9])(?!\d)')
    )


def parse_hour(text):
    """

    :param text:
    :return:

    >>> parse_hour("10点50")
    10

    >>> parse_hour("周19点50")
    9

    >>> parse_hour("0点50")
    0

    """

    return parse_number_with_regex(
        text,
        re.compile(f"(?!{WEEK_PREFIX})([0-2]?[0-9])(?=([点时]))")
    )


def parse_minute(text):
    """

    :param text:
    :return:

    >>> parse_minute("1点10分")
    10

    >>> parse_minute("1点60分")

    >>> parse_minute("1时0分")
    0

    >>> parse_minute("25分")
    25

    >>> parse_minute("1点25")
    25

    >>> parse_minute("1点过25分")
    25

    >>> parse_minute("1点过25")
    25

    >>> parse_minute("1点3刻")
    45

    >>> parse_minute("1时25")
    25

    >>> parse_minute("1点半")
    30

    """

    pat = re.compile(r'(?<=\d[点时])半')
    if pat.search(text):
        return 30

    quarter = parse_number_with_regex(
        text,
        re.compile('(?<=[点时过])[13](?=刻)')
    )

    if quarter is not None:
        return quarter * 15

    return parse_number_with_regex(
        text,
        re.compile('([0-5]?[0-9](?=分(?!钟)))|((?<=(?<!小)[点时过])[0-6]?[0-9](?!刻))'),
        range=[0, 59]
    )


def parse_quarter(text):
    """

    :param text:
    :return:

    >>> parse_quarter("1点3刻")
    3

    >>> parse_quarter("1点过3刻")
    3

    >>> parse_quarter("1时3刻")
    3

    >>> parse_quarter("1时2刻")

    """
    return parse_number_with_regex(
        text,
        re.compile('(?<=[点时过])[13](?=刻)')
    )


def parse_second(text):
    """

    :param text:
    :return:

    >>> parse_second("1分25秒")
    25

    >>> parse_second("1分25")
    25

    >>> parse_second("1分0秒")
    0

    >>> parse_second("1分60")

    """
    return parse_number_with_regex(
        text,
        re.compile('([0-5]?[0-9](?=秒))|((?<=分)[0-6]?[0-9])'),
        range=[0, 59]
    )


def appearing(text, pattern):
    return pattern.search(text) is not None


def weekday_offset(base, weekday, delta_weeks):
    """
    计算X星期几与今天相差的天数

    :param base:
    :param weekday:
    :param delta_weeks:
    :return:

    >>> weekday_offset(1, 2, 0)
    1
    >>> weekday_offset(1, 2, -1)
    -6
    >>> weekday_offset(1, 2, 1)
    8
    >>> weekday_offset(1, 7, 0)
    6
    >>> weekday_offset(1, 7, -1)
    -1
    >>> weekday_offset(7, 1, 1)
    1
    """
    return weekday - base + delta_weeks*7


class FromPattern:
    def __init__(self, pattern):
        self.pat = re.compile(pattern)


class Appearing(FromPattern):
    def __get__(self, instance: 'RawTimeInfo', owner) -> bool:
        return appearing(instance.expr, self.pat)


class NumberOf(FromPattern):
    def __init__(self, pattern, range=None):
        super(NumberOf, self).__init__(pattern)
        self.range = range

    def __get__(self, instance: 'RawTimeInfo', owner) -> Optional[int]:
        return parse_number_with_regex(instance.expr, self.pat, self.range)


class NumberOfHours(FromPattern):
    """

    >>> p = NumberOfHours('后')
    >>> p.parse("半个钟头后")
    0.5
    >>> p.parse("1个半个钟头后")
    >>> p.parse("1个半钟头后")
    1.5
    >>> p.parse("1个半钟后")
    1.5
    >>> p.parse("1个钟后")
    1
    >>> p.parse("1个钟前")
    >>> NumberOfHours('前').parse("1个钟前")
    1
    """

    def __init__(self, key):
        pat = rf'(?:(\d+)|(?:(\d+)个半(?!个))|(?<!个)半)(?=个?(小时|钟头?)[以之]?{key})'
        super(NumberOfHours, self).__init__(pat)

    def parse(self, expr):
        m = self.pat.search(expr)
        if m is None:
            return None
        txt = m.group()
        if txt == '半':
            return 0.5
        if txt.endswith('个半'):
            n = int(m.group(2)) + 0.5
        else:
            n = int(m.group(1))
        return n

    def __get__(self, instance: 'RawTimeInfo', owner) -> Optional[float]:
        return self.parse(instance.expr)


class RawTimeInfo:
    expr: str

    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    quarter: Optional[int] = None
    second: Optional[int] = None

    weekday: Optional[int] = NumberOf(f'(?:{WEEK_PREFIX})[1-7]')

    #
    this_year: bool = Appearing('今年')
    prev_year: bool = Appearing(r'去年|(?<!\d)年前')
    prev_prev_year: bool = Appearing('前年')
    next_year: bool = Appearing(r'明年|(?<!\d)年后')
    next_next_year: bool = Appearing('后年')
    years_before: Optional[int] = NumberOf(r'\d+(?=年[以之]?前)')
    years_after: Optional[int] = NumberOf(r'\d+(?=年[以之]?后)')

    this_month: bool = Appearing('(这个?|本)月')
    prev_month: bool = Appearing('上个?月')
    prev_prev_month: bool = Appearing('上上个?月')
    next_month: bool = Appearing('下个?月')
    next_next_month: bool = Appearing('下下个?月')
    months_before: Optional[int] = NumberOf(r'\d+(?=个?月[以之]?前)')
    months_after: Optional[int] = NumberOf(r'\d+(?=个?月[以之]?后)')

    this_week: bool = Appearing('(?<!上|下)(这|本)个?(周|星期|礼拜)')
    prev_week: bool = Appearing('(?<!上)上个?(周|星期|礼拜)')
    prev_prev_week: bool = Appearing('上上个?(周|星期|礼拜)')
    next_week: bool = Appearing('(?<!下)下个?(周|星期|礼拜)')
    next_next_week: bool = Appearing('下下个?(周|星期|礼拜)')

    today: bool = Appearing('今(?!年)')
    yesterday: bool = Appearing('昨')
    before_yesterday: bool = Appearing('(?<!大|上)前(天|晚)')
    before_before_yesterday: bool = Appearing('(大|上)前(天|晚)')
    tomorrow: bool = Appearing('明(?!年)')
    after_tomorrow: bool = Appearing('(?<!大)后天')
    after_after_tomorrow: bool = Appearing('大后天')
    days_before: Optional[int] = NumberOf(r'\d+(?=天[以之]?前)')
    days_after: Optional[int] = NumberOf(r'\d+(?=天[以之]?后)')

    hours_before: Optional[float] = NumberOfHours('前')
    hours_after: Optional[float] = NumberOfHours('后')

    minutes_before: Optional[int] = NumberOf(r'\d+(?=分钟[以之]?前)')
    minutes_after: Optional[int] = NumberOf(r'\d+(?=分钟[以之]?后)')

    midnight: bool = Appearing('凌晨|半夜')
    morning: bool = Appearing('早(上|晨|间)|晨间|(今|明|清|一)早|上午')
    noon: bool = Appearing('中午|午间')
    afternoon: bool = Appearing('下午|午后')
    night: bool = Appearing('昨晚|前晚|晚上|夜间|夜里|今晚|明晚|半夜')

    def __init__(self, expr: Text):
        """

        :param expr:
        >>> t = RawTimeInfo('大前天')
        >>> t.ensure_set_fields_only('before_before_yesterday')

        >>> t = RawTimeInfo('上前天')
        >>> t.ensure_set_fields_only('before_before_yesterday')

        >>> t = RawTimeInfo('前天')
        >>> t.ensure_set_fields_only('before_yesterday')

        >>> t = RawTimeInfo('大后天')
        >>> t.ensure_set_fields_only('after_after_tomorrow')

        >>> t = RawTimeInfo('后天')
        >>> t.ensure_set_fields_only('after_tomorrow')

        >>> t = RawTimeInfo('今天')
        >>> t.ensure_set_fields_only('today')

        >>> t = RawTimeInfo('今年')
        >>> t.ensure_set_fields_only('this_year')

        >>> t = RawTimeInfo('5天前')
        >>> t.ensure_set_fields_only('days_before')
        >>> t.days_before
        5

        >>> t = RawTimeInfo('5天以后')
        >>> t.ensure_set_fields_only('days_after')
        >>> t.days_after
        5

        >>> t = RawTimeInfo('这周')
        >>> t.ensure_set_fields_only('this_week')

        >>> t = RawTimeInfo('这个周7')
        >>> t.ensure_set_fields_only('this_week', 'weekday')
        >>> t.weekday
        7

        >>> t = RawTimeInfo('上周')
        >>> t.ensure_set_fields_only('prev_week')

        >>> t = RawTimeInfo('上上周')
        >>> t.ensure_set_fields_only('prev_prev_week')

        >>> t = RawTimeInfo('下周')
        >>> t.ensure_set_fields_only('next_week')

        >>> t = RawTimeInfo('下下周')
        >>> t.ensure_set_fields_only('next_next_week')

        >>> t = RawTimeInfo('下下星期1')
        >>> t.ensure_set_fields_only('next_next_week', 'weekday')
        >>> t.weekday
        1

        >>> t = RawTimeInfo('年后1月份')
        >>> t.ensure_set_fields_only('next_year', 'month')
        >>> t.month
        1

        >>> t = RawTimeInfo('5年后1月份')
        >>> t.ensure_set_fields_only('years_after', 'month')
        >>> t.years_after
        5
        >>> t.month
        1

        >>> t = RawTimeInfo('年前')
        >>> t.ensure_set_fields_only('prev_year')
        >>> t.prev_year
        True

        >>> t = RawTimeInfo('1年前')
        >>> t.ensure_set_fields_only('years_before')
        >>> t.years_before
        1

        >>> t = RawTimeInfo('1个小时后')
        >>> t.hours_after
        1

        >>> t = RawTimeInfo('半个小时后')
        >>> t.hours_after
        0.5
        >>> t.ensure_set_fields_only('hours_after')

        >>> t = RawTimeInfo('2个半小时后')
        >>> t.hours_after
        2.5
        >>> t.ensure_set_fields_only('hours_after')

        >>> t = RawTimeInfo('10分钟后')
        >>> t.minutes_after
        10
        >>> t.ensure_set_fields_only('minutes_after')

        >>> t = RawTimeInfo('100分钟后')
        >>> t.minutes_after
        100
        """
        self.expr = expr
        self.year = parse_year(expr)
        self.month = parse_month(expr)
        self.day = parse_day(expr)
        self.hour = parse_hour(expr)
        self.minute = parse_minute(expr)
        self.quarter = parse_quarter(expr)
        self.second = parse_second(expr)

    def ensure_set_fields_only(self, *fields):
        for k, t in self.__annotations__.items():
            if k == 'expr':
                continue
            v = getattr(self, k)
            if t == bool:
                is_set = v
            else:
                is_set = v is not None
            if k in fields:
                assert is_set, f"Unexpected field {k} set to {v}"
            else:
                assert not is_set, f"Expect field {k} to be set"

    def display(self):
        items = [self.expr]
        for k in dir(self):
            if not k.startswith('__') and not callable(getattr(self, k)):
                items.append('='.join([k, str(getattr(self, k))]))
        info = '\n    '.join(items)
        return f'RawTimeInfo<{info}>'


def count_true(*seq) -> int:
    cnt = 0
    for e in seq:
        if e:
            cnt += 1
    return cnt


def at_most_one(*seq) -> bool:
    return count_true(*seq) <= 1


def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


class TimeExpression:
    def __init__(self, expr: str) -> None:
        self.expr = expr


class TimePoint:
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None
    weekday: Optional[int] = None
    ampm: Optional[str] = None

    raw: Optional[RawTimeInfo]
    baseline: datetime

    def __init__(self, time_expr: Union[Text, TimeExpression, None] = None, baseline: datetime = None):
        """

        :param time_expr:
        :param baseline:
        >>> t = TimePoint('周1早上8点')
        >>> t.fuzzy_day
        False
        >>> t.fuzzy_week
        True

        >>> t = TimePoint("今天下午两点十分")
        >>> t.year == t.baseline.year
        True
        >>> t.month == t.baseline.month
        True
        >>> t.day == t.baseline.day
        True
        >>> t.raw.afternoon
        True
        >>> t.hour
        2
        >>> t.computed_hour()
        14
        >>> t.minute
        10
        >>> t.second
        >>> t.fuzzy_apm
        False
        >>> t.fuzzy_day
        False
        >>> t.fuzzy_year
        False
        >>> t.fuzzy_month
        False
        >>> t.fuzzy_week
        False
        """

        # TODO: 可配置时区
        self.baseline = baseline or datetime.now(timezone(timedelta(hours=8)))
        if isinstance(time_expr, TimeExpression):
            self.raw = RawTimeInfo(time_expr.expr)
            self.parse_raw()
        elif isinstance(time_expr, str):
            self.raw = RawTimeInfo(preprocess(time_expr))
            self.parse_raw()
        elif isinstance(time_expr, dict):
            self.load_dict(time_expr)
            self.raw = None
        elif time_expr is None:
            self.raw = None
        else:
            abort('Invalid time_expr type')

    @staticmethod
    def get_fields():
        # return self.__annotations__.keys()
        return ['year', 'month', 'day', 'hour', 'minute', 'second', 'weekday', 'ampm', 'baseline']

    @property
    def fuzzy_year(self):
        return self.year is None and self.month is not None

    @property
    def fuzzy_month(self):
        return self.month is None and self.day is not None

    @property
    def fuzzy_day(self):
        return self.day is None and self.weekday is None and self.hour is not None

    @property
    def fuzzy_week(self):
        return self.day is None and self.weekday is not None

    @property
    def fuzzy_apm(self):
        return self.ampm is None and self.hour is not None and self.hour <= 12

    def parse_raw(self):
        self.fill_year(self.raw)
        self.fill_month(self.raw)
        self.fill_day(self.raw)
        self.fill_ampm(self.raw)
        self.fill_hour(self.raw)
        self.fill_minute(self.raw)
        self.fill_second(self.raw)
        self.validate()

    def load_dict(self, info: Dict[Text, Any]):
        info = info.copy()
        baseline = info.pop('baseline', None)
        if baseline:
            self.baseline = datetime.strptime(baseline, "%Y-%m-%d %H:%M:%S")
        for k in self.get_fields():
            v = info.get(k)
            if v is not None:
                setattr(self, k, v)

    def dump_to_dict(self):
        rv = {k: getattr(self, k) for k in self.get_fields() if k != 'baseline'}
        rv['baseline'] = self.baseline.strftime('%Y-%m-%d %H:%M:%S')
        return rv

    def update(self, info: 'TimePoint'):
        """

        :param info:
        :return:

        >>> baseline = datetime(2000, 1, 1)
        >>> t0 = TimePoint("明天早上", baseline)
        >>> t1 = TimePoint("7点", baseline)
        >>> t0.get_datetime_str()
        '2000-01-02 08:00:00'
        >>> t0.update(t1)
        >>> t0.get_datetime_str()
        '2000-01-02 07:00:00'

        >>> baseline = datetime(2000, 1, 1)
        >>> t0 = TimePoint("明天早上", baseline)
        >>> t1 = TimePoint("周一", baseline)
        >>> t0.get_datetime_str(True)
        '2000-01-02 08:00:00'
        >>> t0.update(t1)
        >>> t0.get_datetime_str(True)
        '2000-01-03 08:00:00'
        """
        if info.day is not None:
            self.weekday = None
        if info.weekday is not None:
            self.year = None
            self.month = None
            self.day = None
        for key in self.get_fields():
            new_value = getattr(info, key)
            if new_value is not None:
                setattr(self, key, new_value)

    def fill_year(self, info: RawTimeInfo):
        """

        :param info:
        :return:
        >>> t = TimePoint("今年")
        >>> t.year == t.baseline.year
        True
        >>> t.month
        >>> t.day
        >>> t.fuzzy_year
        False

        >>> t = TimePoint("去年")
        >>> t.year == t.baseline.year - 1
        True

        """
        cnt = count_true(
            info.year is not None,
            info.this_year,
            info.prev_year,
            info.prev_prev_year,
            info.next_year,
            info.next_next_year,
            info.years_before is not None,
            info.years_after is not None,
            )

        if cnt == 0:
            return

        if cnt > 1:
            abort('Invalid year info')

        if info.year is not None:
            self.year = info.year
        elif info.this_year:
            self.year = self.baseline.year
        elif info.prev_year:
            self.year = self.baseline.year - 1
        elif info.next_year:
            self.year = self.baseline.year + 1
        elif info.prev_prev_year:
            self.year = self.baseline.year - 2
        elif info.next_next_year:
            self.year = self.baseline.year + 2
        elif info.years_before is not None:
            # TODO: set month and day?
            self.year = self.baseline.year - info.years_before
        elif info.years_after is not None:
            self.year = self.baseline.year + info.years_after
        else:
            assert unreachable

    def fill_month(self, info: RawTimeInfo):
        """

        :param info:
        :return:

        >>> t = TimePoint("下个月", baseline=datetime.strptime("2000-12-01", "%Y-%m-%d"))
        >>> [t.year, t.month, t.day]
        [2001, 1, None]

        >>> t = TimePoint("上个月", baseline=datetime.strptime("2000-01-01", "%Y-%m-%d"))
        >>> [t.year, t.month, t.day]
        [1999, 12, None]

        >>> t = TimePoint("5月")
        >>> [t.year, t.month, t.day]
        [None, 5, None]
        >>> t.fuzzy_year
        True

        >>> t = TimePoint("一个月后", baseline=datetime.strptime("2000-01-30", "%Y-%m-%d"))
        >>> [t.year, t.month, t.day, t.hour, t.minute]
        [2000, 2, 29, None, None]

        """
        cnt = count_true(
            info.month is not None,
            info.this_month,
            info.prev_month,
            info.prev_prev_month,
            info.next_month,
            info.next_next_month,
            info.months_before is not None,
            info.months_after is not None,
        )
        if cnt == 0:
            return

        if cnt > 1:
            abort('Invalid month info')

        if info.month is not None:
            self.month = info.month
        else:
            delta = None
            if info.this_month:
                delta = 0
            elif info.prev_month:
                delta = -1
            elif info.prev_prev_month:
                delta = -2
            elif info.next_month:
                delta = 1
            elif info.next_next_month:
                delta = 2

            if delta is not None:
                point = add_months(self.baseline, delta)
                self.year = point.year
                self.month = point.month
            else:
                if info.months_before is not None:
                    delta = -info.months_before
                elif info.months_after is not None:
                    delta = info.months_after
                else:
                    assert unreachable
                    delta = 0
                self.set_date(add_months(self.baseline, delta))

    def fill_day(self, info: RawTimeInfo):
        """

        :param info:
        :return:
        >>> t = TimePoint("后天", baseline=datetime.strptime("1999-12-31", "%Y-%m-%d"))
        >>> [t.year, t.month, t.day, t.hour, t.minute]
        [2000, 1, 2, None, None]

        >>> t = TimePoint("大后天", baseline=datetime.strptime("1999-12-31", "%Y-%m-%d"))
        >>> [t.year, t.month, t.day, t.hour, t.minute]
        [2000, 1, 3, None, None]

        >>> t = TimePoint("三天后", baseline=datetime.strptime("1999-12-31", "%Y-%m-%d"))
        >>> [t.year, t.month, t.day, t.hour, t.minute]
        [2000, 1, 3, None, None]

        >>> t = TimePoint("三天前", baseline=datetime.strptime("2000-01-01", "%Y-%m-%d"))
        >>> [t.year, t.month, t.day, t.hour, t.minute]
        [1999, 12, 29, None, None]

        >>> t = TimePoint("明天", baseline=datetime.strptime("1999-12-31", "%Y-%m-%d"))
        >>> [t.year, t.month, t.day, t.hour, t.minute]
        [2000, 1, 1, None, None]

        >>> t = TimePoint("昨天", baseline=datetime.strptime("2000-01-01", "%Y-%m-%d"))
        >>> [t.year, t.month, t.day, t.hour, t.minute]
        [1999, 12, 31, None, None]

        >>> t = TimePoint("周末")
        >>> t.fuzzy_week
        True
        >>> t.weekday
        7
        >>> t = TimePoint("这周下周周末")
        Traceback (most recent call last):
            ...
        ParseTimeError: Only one week modifier allowed

        >>> t = TimePoint("周一", datetime(2019, 8, 30, 11, 18))
        >>> t.get_datetime_str(True)
        '2019-09-02 08:00:00'
        """
        cnt = count_true(
            info.day is not None or info.weekday is not None,
            info.today,
            info.yesterday,
            info.before_yesterday,
            info.before_before_yesterday,
            info.tomorrow,
            info.after_tomorrow,
            info.after_after_tomorrow,
            info.days_before is not None,
            info.days_after is not None,
        )
        if cnt == 0:
            return

        if cnt > 1:
            abort('Invalid day info')

        if info.day is not None:
            self.day = info.day
        elif info.weekday is not None:
            valid = at_most_one(
                info.this_week,
                info.prev_week,
                info.next_week,
                info.next_next_week,
                info.prev_prev_week,
            )
            if not valid:
                abort('Only one week modifier allowed')

            self.weekday = info.weekday

            if info.this_week:
                week = 0
            elif info.prev_week:
                week = -1
            elif info.next_week:
                week = 1
            elif info.prev_prev_week:
                week = -2
            elif info.next_next_week:
                week = 2
            else:
                return

            delta_days = weekday_offset(self.baseline.weekday()+1, info.weekday, week)
            self.set_date(self.baseline + timedelta(days=delta_days))

        elif info.today:
            self.set_date(self.baseline)
        elif info.tomorrow:
            self.set_date(self.baseline + timedelta(days=1))
        elif info.after_tomorrow:
            self.set_date(self.baseline + timedelta(days=2))
        elif info.after_after_tomorrow:
            self.set_date(self.baseline + timedelta(days=3))
        elif info.yesterday:
            self.set_date(self.baseline + timedelta(days=-1))
        elif info.before_yesterday:
            self.set_date(self.baseline + timedelta(days=-2))
        elif info.before_before_yesterday:
            self.set_date(self.baseline + timedelta(days=-3))
        elif info.days_before:
            self.set_date(self.baseline + timedelta(days=-info.days_before))
        elif info.days_after:
            self.set_date(self.baseline + timedelta(days=info.days_after))
        else:
            assert unreachable

    def fill_ampm(self, info: RawTimeInfo):
        cnt = count_true(
            info.midnight,
            info.morning,
            info.noon,
            info.afternoon,
            info.night
        )
        if cnt > 1:
            abort('Too many am/pm qualifiers')

        self.ampm = {
            info.midnight: 'midnight',
            info.morning: 'morning',
            info.noon: 'noon',
            info.afternoon: 'afternoon',
            info.night: 'night'
        }.get(True)

    def fill_hour(self, info: RawTimeInfo):
        """

        :param info:
        :return:

        >>> t = TimePoint("凌晨五点")
        >>> t.year, t.month, t.day, t.hour, t.minute
        (None, None, None, 5, None)

        >>> t = TimePoint("下午五点")
        >>> t.year, t.month, t.day, t.computed_hour(), t.minute
        (None, None, None, 17, None)

        >>> t = TimePoint("晚上五点")
        >>> t.fuzzy_day, t.fuzzy_apm, t.computed_hour(), t.minute
        (True, False, 17, None)

        >>> t = TimePoint("中午1点半")
        >>> t.computed_hour(), t.minute
        (13, 30)

        >>> t = TimePoint("中午12点半")
        >>> t.computed_hour(), t.minute
        (12, 30)

        >>> t = TimePoint("晚上12点半")
        >>> t.computed_hour(), t.minute
        (0, 30)

        >>> t = TimePoint("晚上11点半")
        >>> t.computed_hour(), t.minute
        (23, 30)

        >>> t = TimePoint("晚上1点")
        >>> t.computed_hour(), t.minute
        (1, None)

        >>> t = TimePoint("一个小时后", baseline=datetime.strptime("1999-12-31 23:59:59", "%Y-%m-%d %H:%M:%S"))
        >>> t.year, t.month, t.day, t.computed_hour(), t.minute, t.second
        (2000, 1, 1, 0, 59, 59)

        >>> t = TimePoint("一个小时之后", baseline=datetime.strptime("1999-12-31 23:59:59", "%Y-%m-%d %H:%M:%S"))
        >>> t.year, t.month, t.day, t.computed_hour(), t.minute, t.second
        (2000, 1, 1, 0, 59, 59)
        """
        cnt = count_true(
            info.hour is not None,
            info.hours_before is not None,
            info.hours_after is not None,
        )
        if cnt == 0:
            return
        if cnt > 1:
            abort('Invalid hour info')

        if info.hours_after is not None:
            self.set_datetime(self.baseline + timedelta(hours=info.hours_after))
        elif info.hours_before is not None:
            self.set_datetime(self.baseline - timedelta(hours=info.hours_before))
        elif info.hour is not None:
            self.hour = info.hour
        else:
            assert unreachable

    def fill_minute(self, info: RawTimeInfo):
        """

        :param info:
        :return:

        >>> t = TimePoint("一点三刻")
        >>> t.fuzzy_day, t.fuzzy_apm, t.hour, t.minute
        (True, True, 1, 45)

        >>> t = TimePoint("五分钟后", baseline=datetime.strptime("01:59:01", "%H:%M:%S"))
        >>> t.hour, t.minute, t.second
        (2, 4, 1)

        >>> t = TimePoint("五分钟前", baseline=datetime.strptime("01:00:01", "%H:%M:%S"))
        >>> t.hour, t.minute, t.second
        (0, 55, 1)

        >>> t = TimePoint("五分钟之后", baseline=datetime.strptime("01:59:01", "%H:%M:%S"))
        >>> t.hour, t.minute, t.second
        (2, 4, 1)

        """
        cnt = count_true(
            info.minute is not None,
            info.minutes_after is not None,
            info.minutes_before is not None,
        )
        if cnt == 0:
            return

        if cnt > 1:
            abort('Invalid minute')

        if info.minutes_after is not None:
            self.set_datetime(self.baseline + timedelta(minutes=info.minutes_after))
        elif info.minutes_before is not None:
            self.set_datetime(self.baseline - timedelta(minutes=info.minutes_before))
        elif info.minute is not None:
            if self.hour is None:
                abort('Standalone minute means nothing')
            self.minute = info.minute
        else:
            assert unreachable

    def fill_second(self, info: RawTimeInfo):
        if info.second is not None:
            self.second = info.second

    def validate(self):
        """
        >>> t = TimePoint("明年五点")
        Traceback (most recent call last):
            ...
        ParseTimeError: Time expression must not has a gap
        """

        # 时间阶梯必须连续, 中间不能有空洞，如："下个月五点"是错误的
        data = [None, self.year, self.month, self.day, self.hour, self.minute, self.second]
        data = [x is not None for x in data]
        cnt = 0
        for i in range(len(data)-1):
            if data[i:i+2] == [False, True]:
                cnt += 1
        if cnt > 1:
            abort('Time expression must not has a gap')

    def computed_hour(self):
        if self.hour is None:
            return None
        if self.hour > 12 or self.ampm is None:
            return self.hour

        hour = self.hour
        if self.ampm == 'midnight':
            if hour >= 10:
                hour += 12
        elif self.ampm == 'noon':
            if hour < 5:
                hour += 12
        elif self.ampm == 'afternoon':
            hour += 12
        elif self.ampm == 'night':
            if hour > 4:
                hour += 12
        elif self.ampm == 'morning':
            pass
        else:
            pass
        hour %= 24
        return hour

    def fill_apm(self, info: RawTimeInfo):
        if all([
            self.year is None,
            self.month is None,
            self.day is None,
            self.hour is None,
            self.minute is None,
            self.second is None,
            self.weekday is None,
        ]):
            if info.midnight:
                self.set_date(self.baseline + timedelta(days=1))
            elif any([info.morning, info.noon, info.afternoon, info.night]):
                self.set_date(self.baseline)

    def get_datetime_str(self, prefer_future=False):
        return self.get_datetime(prefer_future).strftime("%Y-%m-%d %H:%M:%S")

    def get_datetime(self, prefer_future=False):
        """
        >>> TimePoint("5月", baseline=datetime(year=2000, month=1, day=1)).get_datetime_str()
        '2000-05-01 08:00:00'

        >>> TimePoint("3月", baseline=datetime(year=2000, month=4, day=1)).get_datetime_str()
        '2000-03-01 08:00:00'

        >>> TimePoint("5号", baseline=datetime(year=2000, month=4, day=10)).get_datetime_str()
        '2000-04-05 08:00:00'

        >>> TimePoint("5号", baseline=datetime(year=2000, month=4, day=25)).get_datetime_str()
        '2000-05-05 08:00:00'

        >>> TimePoint("27号", baseline=datetime(year=2000, month=4, day=25)).get_datetime_str()
        '2000-04-27 08:00:00'

        >>> TimePoint("27号", baseline=datetime(year=2000, month=4, day=25)).get_datetime_str()
        '2000-04-27 08:00:00'

        >>> TimePoint("周一", baseline=datetime(year=2000, month=1, day=1)).get_datetime_str()  # 2000-01-01 星期6
        '2000-01-03 08:00:00'

        >>> TimePoint("周2", baseline=datetime(year=2000, month=1, day=8)).get_datetime_str()
        '2000-01-04 08:00:00'

        >>> TimePoint("两点钟", baseline=datetime(year=2000, month=1, day=1, hour=11)).get_datetime_str()
        '2000-01-01 14:00:00'

        >>> TimePoint("上午9点", baseline=datetime(year=2000, month=1, day=1, hour=11)).get_datetime_str(True)
        '2000-01-02 09:00:00'

        >>> TimePoint("周天", baseline=datetime(year=2000, month=1, day=1)).get_datetime_str()  # 2000-01-01 星期6
        '2000-01-02 08:00:00'

        """
        if not at_most_one(
                self.fuzzy_year,
                self.fuzzy_month,
                self.fuzzy_week,
                self.fuzzy_day,
        ):
            abort(f'Too many fuzzy: {self.get_fuzzies()}')

        def get_nearest(points):
            deltas = [p - self.baseline for p in points]
            _, the_nearest = min([(abs(d), i) for i, d in enumerate(deltas)])
            return points[the_nearest]

        year = self.year or self.baseline.year
        month = self.month or 1
        day = self.day or 1
        hour = self.computed_hour() or 8
        minute = self.minute or 0
        second = self.second or 0

        day_ = day or self.baseline.day

        if self.fuzzy_year:
            points = [
                datetime(year=self.baseline.year-1, month=self.month, day=day_),
                datetime(year=self.baseline.year, month=self.month, day=day_),
                datetime(year=self.baseline.year+1, month=self.month, day=day_),
            ]
            nearest = get_nearest(points)
            year = nearest.year
            if prefer_future and nearest < self.baseline:
                year += 1

        elif self.fuzzy_month:
            month = self.baseline.month
            points = [
                datetime(year=self.baseline.year, month=month-1, day=day_),
                datetime(year=self.baseline.year, month=month, day=day_),
                datetime(year=self.baseline.year, month=month+1, day=day_),
            ]
            nearest = get_nearest(points)

            if prefer_future and nearest < self.baseline:
                nearest = add_months(nearest, 1)

            year = nearest.year
            month = nearest.month
        elif self.fuzzy_week:
            today = self.baseline.weekday() + 1
            delta = self.weekday - today
            if delta <= 0:
                if prefer_future:
                    delta += 7
                elif today in (5, 6, 7) and self.weekday == 1:
                    # 周五六日说周一通常指下周一
                    delta += 7
            point = self.baseline + timedelta(days=delta)
            year = point.year
            month = point.month
            day = point.day
        elif self.fuzzy_day:
            # 没说上下午的情况下，如果时间和当前时间靠近，则倾向于就近的时间
            if self.fuzzy_apm:
                if hour + 12 - self.baseline.hour < 6:
                    hour += 12

            delta = hour - self.baseline.hour
            if prefer_future and delta < 0:
                delta += 24
            point = self.baseline + timedelta(hours=delta)
            year = point.year
            month = point.month
            day = point.day

        return datetime(year, month, day, hour, minute, second)

    def set_date(self, dt):
        self.day = dt.day
        self.year = dt.year
        self.month = dt.month

    def set_datetime(self, dt):
        self.set_date(dt)
        self.hour = dt.hour
        self.minute = dt.minute
        self.second = dt.second

    def get_fuzzies(self):
        fuzzies = []
        for k in dir(self):
            if k.startswith('fuzzy_'):
                v = getattr(self, k)
                if v:
                    fuzzies.append(k)
        return ','.join(fuzzies)

    def __repr__(self):
        flags = self.get_fuzzies()
        ts = f'{self.year}-{self.month}-{self.day} {self.hour}:{self.minute}:{self.second} {self.weekday}'
        contents = [ts, self.raw and self.raw.expr or 'N/A']
        if flags:
            contents.insert(1, flags)
        contents = ' '.join(contents)
        try:
            guess = self.get_datetime_str()
        except ParseTimeError as e:
            guess = str(e)
        return f'TimePoint<{guess}|{contents}>'

    def __add__(self, other):
        """

        :param other:
        :return:

        >>> baseline = datetime(2000, 1, 1, 1, 1, 1)
        >>> t = TimePoint("今天", baseline) + TimePoint("下午") + TimePoint('8点') + TimePoint('明天', baseline)
        >>> [t.year, t.month, t.day, t.hour, t.minute, t.second, t.ampm]
        [2000, 1, 2, 8, None, None, 'afternoon']
        >>> t.get_datetime_str()
        '2000-01-02 20:00:00'
        """
        rv = TimePoint()
        rv.update(self)
        rv.update(other)
        rv.baseline = self.baseline
        return rv


if __name__ == '__main__':
    import doctest
    doctest.testmod()
