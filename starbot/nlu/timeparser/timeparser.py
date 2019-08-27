import re
import datetime

from typing import Text, Optional

if __name__ == '__main__':
    from numberify import numberify, WEEK_PREFIX
    from pattern import patterns
else:
    from starbot.nlu.timeparser.numberify import numberify, WEEK_PREFIX
    from starbot.nlu.timeparser.pattern import patterns


def preprocess(text):
    p = re.compile(r'\s+|[的]+')
    text = p.sub('', text)
    text = numberify(text)
    return text


def get_time_tokens(text):
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
    return tokens


def parse_all_time(text):
    for token in get_time_tokens(text):
        yield TimePoint(token)


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
        re.compile('((?<!\d))([0-3][0-9]|[1-9])(?=([日号]))|(?<=月)([0-3][0-9]|[1-9])(?!\d)')
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

    >>> parse_minute("1时25")
    25

    """
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


def parse_seconds(text):
    """

    :param text:
    :return:

    >>> parse_seconds("1分25秒")
    25

    >>> parse_seconds("1分25")
    25

    >>> parse_seconds("1分0秒")
    0

    >>> parse_seconds("1分60")

    """
    return parse_number_with_regex(
        text,
        re.compile('([0-5]?[0-9](?=秒))|((?<=分)[0-6]?[0-9])'),
        range=[0, 59]
    )


def appearing(text, pattern):
    return pattern.search(text) is not None


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
    def __init__(self, key):
        """

        :param key:

        >>> p = NumberOfHours('后')
        >>> p.parse("半个钟头后")
        0.5
        >>> p.parse("一个半个钟头后")
        >>> p.parse("一个半钟头后")
        1.5
        >>> p.parse("一个半钟后")
        1.5
        >>> p.parse("一个钟后")
        1
        >>> p.parse("一个钟前")
        >>> NumberOfHours('前').parse("一个钟前")
        1
        """
        pat = '(?:(\d+)|(?:(\d+)个半(>!个))|半)(?=个?(小时|钟头?)[以之]?){key}'
        super(NumberOfHours, self).__init__(pat)

    def parse(self, expr):
        m = self.pat.search(expr)
        if m is None:
            return None
        txt = m.group()
        if txt == '半':
            return 0.5
        n = int(m.group(1))
        if txt.endswith('个半'):
            n += 0.5
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
    seconds: Optional[int] = None

    week_day: Optional[int] = NumberOf(f'(?:{WEEK_PREFIX})[0-6]')

    #
    this_year: bool = Appearing('今年')
    prev_year: bool = Appearing('去年|(?<!\d)年前')
    prev_prev_year: bool = Appearing('前年')
    next_year: bool = Appearing('明年|(?<!\d)年后')
    next_next_year: bool = Appearing('后年')
    years_before: Optional[int] = NumberOf('\d+(?=年[以之]?前)')
    years_after: Optional[int] = NumberOf('\d+(?=年[以之]?后)')

    this_month: bool = Appearing('(这个?|本)月')
    prev_month: bool = Appearing('上个?月')
    prev_prev_month: bool = Appearing('上上个?月')
    next_month: bool = Appearing('下个?月')
    next_next_month: bool = Appearing('下下个?月')
    months_before: Optional[int] = NumberOf('\d+(?=个?月[以之]?前)')
    months_after: Optional[int] = NumberOf('\d+(?=个?月[以之]?后)')

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
    days_before: Optional[int] = NumberOf('\d+(?=天[以之]?前)')
    days_after: Optional[int] = NumberOf('\d+(?=天[以之]?后)')

    hours_before: Optional[float] = NumberOfHours('前')
    hours_after: Optional[float] = NumberOfHours('后')

    minutes_before: Optional[int] = NumberOf('\d+(?=分钟[以之]?前)')
    minutes_after: Optional[int] = NumberOf('\d+(?=分钟[以之]?后)')

    early_morning: bool = Appearing('凌晨|半夜')
    morning: bool = Appearing('早(上|晨|间)|晨间|今早|明早|上午')
    noon: bool = Appearing('中午|午间')
    afternoon: bool = Appearing('下午|午后')
    night: bool = Appearing('前晚|晚上|夜间|夜里|今晚|明晚|半夜')

    def __init__(self, expr: Text):
        """

        :param expr:
        >>> t = RawTimeInfo('大前天')
        >>> assert not t.yesterday
        >>> assert not t.before_yesterday
        >>> assert t.before_before_yesterday

        >>> t = RawTimeInfo('上前天')
        >>> assert not t.yesterday
        >>> assert not t.before_yesterday
        >>> assert t.before_before_yesterday

        >>> t = RawTimeInfo('前天')
        >>> assert not t.yesterday
        >>> assert t.before_yesterday
        >>> assert not t.before_before_yesterday

        >>> t = RawTimeInfo('大后天')
        >>> assert not t.tomorrow
        >>> assert not t.after_tomorrow
        >>> assert t.after_after_tomorrow

        >>> t = RawTimeInfo('后天')
        >>> assert not t.tomorrow
        >>> assert t.after_tomorrow
        >>> assert not t.after_after_tomorrow

        >>> t = RawTimeInfo('今天')
        >>> t.today
        True
        >>> t.this_year
        False

        >>> t = RawTimeInfo('今年')
        >>> t.today
        False
        >>> t.this_year
        True

        >>> t = RawTimeInfo('5天前')
        >>> t.days_after
        >>> t.days_before
        5
        >>> t.years_before
        >>> t.years_after

        >>> t = RawTimeInfo('5天以后')
        >>> t.days_after
        5
        >>> t.days_before
        >>> t.years_before
        >>> t.years_after

        >>> t = RawTimeInfo('这周')
        >>> assert t.this_week
        >>> assert not t.prev_week
        >>> assert not t.prev_prev_week
        >>> assert not t.next_week
        >>> assert not t.next_next_week

        >>> t = RawTimeInfo('这个周末')
        >>> assert t.this_week
        >>> assert not t.prev_week
        >>> assert not t.prev_prev_week
        >>> assert not t.next_week
        >>> assert not t.next_next_week

        >>> t = RawTimeInfo('上周')
        >>> assert not t.this_week
        >>> assert t.prev_week
        >>> assert not t.prev_prev_week
        >>> assert not t.next_week
        >>> assert not t.next_next_week

        >>> t = RawTimeInfo('上上周')
        >>> assert not t.this_week
        >>> assert not t.prev_week
        >>> assert t.prev_prev_week
        >>> assert not t.next_week
        >>> assert not t.next_next_week

        >>> t = RawTimeInfo('下周')
        >>> assert not t.this_week
        >>> assert not t.prev_week
        >>> assert not t.prev_prev_week
        >>> assert t.next_week
        >>> assert not t.next_next_week

        >>> t = RawTimeInfo('下下周')
        >>> assert not t.this_week
        >>> assert not t.prev_week
        >>> assert not t.prev_prev_week
        >>> assert not t.next_week
        >>> assert t.next_next_week

        >>> t = RawTimeInfo('下下星期一')
        >>> assert not t.this_week
        >>> assert not t.prev_week
        >>> assert not t.prev_prev_week
        >>> assert not t.next_week
        >>> assert t.next_next_week

        >>> t = RawTimeInfo('年后1月份')
        >>> t.next_year
        True
        >>> t.month
        1
        >>> t.years_after

        >>> t = RawTimeInfo('5年后1月份')
        >>> t.years_after
        5
        >>> t.next_year
        False
        >>> t.month
        1

        >>> t = RawTimeInfo('年前')
        >>> t.prev_year
        True
        >>> t.years_before

        >>> t = RawTimeInfo('1年前')
        >>> t.prev_year
        False
        >>> t.years_before
        1

        >>> t = RawTimeInfo('1个小时后')
        >>> t.hours_after
        1

        >>> t = RawTimeInfo('半个小时后')
        >>> t.hours_after
        0.5

        >>> t = RawTimeInfo('2个半小时后')
        >>> t.hours_after
        2.5
        """
        self.expr = expr
        self.year = parse_year(expr)
        self.month = parse_month(expr)
        self.day = parse_day(expr)
        self.hour = parse_hour(expr)
        self.minute = parse_minute(expr)
        self.quarter = parse_quarter(expr)
        self.seconds = parse_seconds(expr)

    def display(self):
        items = [self.expr]
        for k in dir(self):
            if not k.startswith('__') and not callable(getattr(self, k)):
                items.append('='.join([k, str(getattr(self, k))]))
        info = '\n    '.join(items)
        return f'RawTimeInfo<{info}>'


def only_one(*seq) -> bool:
    cnt = 0
    for e in seq:
        if e:
            cnt += 1
    return cnt == 1


class TimePoint:
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    seconds: Optional[int] = None

    raw: RawTimeInfo
    baseline: datetime.datetime

    def __init__(self, time_expr: Text):
        self.raw = RawTimeInfo(time_expr)
        self.baseline = datetime.datetime.now()
        self.adjust_year(self.raw)
        self.adjust_month(self.raw)
        self.adjust_day(self.raw)

    def adjust_year(self, info: RawTimeInfo):
        valid = only_one(
            info.year is not None,
            info.this_year,
            info.prev_year,
            info.prev_prev_year,
            info.next_year,
            info.next_next_year,
            info.years_before is not None,
            info.years_after is not None,
            )

        if valid:
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
                self.year = self.baseline.year - info.years_before
            elif info.years_after is not None:
                self.year = self.baseline.year + info.years_after
            else:
                assert not 'reachable'

    def adjust_month(self, info: RawTimeInfo):
        valid = only_one(
            info.month is not None,
            info.this_month,
            info.prev_month,
            info.prev_prev_month,
            info.next_month,
            info.next_next_month,
            info.months_before is not None,
            info.months_after is not None,
        )

        if valid:
            if info.month is not None:
                self.month = info.month
            elif info.this_month:
                self.year = self.baseline.year
                self.month = self.baseline.month
            elif info.prev_month:
                self.year = self.baseline.year
                self.month = self.baseline.month - 1
            elif info.prev_prev_month:
                self.year = self.baseline.year
                self.month = self.baseline.month - 2
            elif info.next_month:
                self.year = self.baseline.year
                self.month = self.baseline.month + 1
            elif info.next_next_month:
                self.year = self.baseline.year
                self.month = self.baseline.month + 2
            elif info.months_before is not None:
                self.year = self.baseline.year
                self.month = self.baseline.month - info.months_before
                self.day = self.baseline.day
            elif info.months_after is not None:
                self.year = self.baseline.year
                self.month = self.baseline.month + info.months_after
                self.day = self.baseline.day
            else:
                assert not 'reachable'

            if self.month > 12:
                self.month -= 12
                self.year -= 1
            if self.month < 0:
                self.month += 12
                self.year += 1

    def adjust_day(self, info: RawTimeInfo):
        valid = only_one(
            info.day is not None or info.week_day is not None,
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
        if valid:
            if info.day is not None:
                self.day = info.day
            elif info.week_day is not None:
                # TODO
                if info.this_week:
                    pass
            elif info.today:
                self.set_date(self.baseline)
            elif info.tomorrow:
                self.set_date(self.baseline + datetime.timedelta(days=1))
            elif info.after_tomorrow:
                self.set_date(self.baseline + datetime.timedelta(days=2))
            elif info.after_after_tomorrow:
                self.set_date(self.baseline + datetime.timedelta(days=3))
            elif info.yesterday:
                self.set_date(self.baseline + datetime.timedelta(days=-1))
            elif info.before_yesterday:
                self.set_date(self.baseline + datetime.timedelta(days=-2))
            elif info.before_before_yesterday:
                self.set_date(self.baseline + datetime.timedelta(days=-3))
            elif info.days_before:
                self.set_date(self.baseline + datetime.timedelta(days=-info.days_before))
            elif info.days_after:
                self.set_date(self.baseline + datetime.timedelta(days=info.days_after))
            else:
                assert not 'reachable'

    def set_date(self, dt):
        self.day = dt.day
        self.year = dt.year
        self.month = dt.month



if __name__ == '__main__':
    # list(parse_all_time("明天上午十点去打球后天去旅行"))
    import doctest
    doctest.testmod()
