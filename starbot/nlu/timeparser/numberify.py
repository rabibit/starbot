import re
import math
from typing import Text


DIGITS = '一二两三四五六七八九123456789'
UNITS = "亿万千百十一"
VOLUMES = [1_0000_0000, 10000, 1000, 100, 10, 1]

WEEK_PREFIX = '(?<=周)|(?<=(星期|礼拜))'


def reduce(text: Text, unit: Text):
    """

    :param text:
    :param unit:
    :return:

    >>> reduce('1万3千05十', '十')
    '1万3千50'
    >>> reduce('1万3千50', '百')
    '1万3千50'
    >>> reduce('1万3千50', '千')
    '1万3050'
    >>> reduce('10万3050', '万')
    '103050'
    >>> reduce('周7十点', '十')
    '周710点'
    >>> reduce('明天十点', '十')
    '明天10点'
    """
    index = UNITS.find(unit)
    vol = VOLUMES[index]
    postfix = '[0-9]{0,%s}' % round(math.log(vol, 10))
    prefix = '[0-9]+' if unit in '万亿' else '0?[1-9]?'
    if unit == '十':
        # to avoid 周7十点 to be translated to 周70点
        prefix = f'(?!{WEEK_PREFIX})' + prefix
    regex = f'{prefix}{unit}{postfix}'

    p = re.compile(regex)

    def repl(m):
        s = m.group().split(unit)
        high = int(s[0] or 1)
        low = int(s[1] or 0)
        return str(high*vol + low)

    return p.sub(repl, text)


def process_single_digit(text):
    """

    :param text:
    :return:
    >>> process_single_digit("一")
    '1'
    >>> process_single_digit("一万四千五")
    '1万4千5'
    >>> process_single_digit("这周天下个礼拜日")
    '这周7下个礼拜7'
    """
    p = re.compile(f"[零一二两三四五六七八九]|({WEEK_PREFIX})[末天日]")
    return p.sub(lambda m: str(word2num(m.group())), text)


def process_simple_2parts(text: Text, unit: Text):
    """
    处理万百千简写的数字

    :param text:
    :param unit:
    :return:

    >>> process_simple_2parts('一万五', '万')
    '15000'
    >>> process_simple_2parts('一万五', '千')
    '一万五'
    >>> process_simple_2parts('一万五十', '万')
    '一万五十'
    >>> process_simple_2parts('五十五', '十')
    '55'
    """
    index = UNITS.find(unit)
    remain = UNITS[index+1:]
    regex = f'[{DIGITS}]{unit}[{DIGITS}]'
    if remain:
        regex += f'(?!([{remain}]))'
    p = re.compile(regex)

    def repl(m):
        s = m.group().split(unit)
        num = 0
        if len(s) == 2:
            num += word2num(s[0])*VOLUMES[index] + word2num(s[1])*VOLUMES[index+1]
        return str(num)
    return p.sub(repl, text)


def numberify(text):
    """

    :param text:
    :return:

    >>> numberify('我要四千个苹果和一万零5个梨')
    '我要4000个苹果和10005个梨'
    >>> numberify('四千三百万五十亿四千三百万五十')
    '4300005043000050'
    >>> numberify('一万五')
    '15000'
    >>> numberify('四万五千六百七十八')
    '45678'
    >>> numberify('4万5千六百七十八')
    '45678'
    >>> numberify('4万零六百七十八')
    '40678'
    >>> numberify('4万五千零六百七十八')
    '45678'
    >>> numberify('4万五千零八')
    '45008'
    >>> numberify('明天上午十点')
    '明天上午10点'
    >>> numberify('周天十点')
    '周010点'
    >>> numberify('周天九点')
    '周09点'
    """
    for unit in '亿万千百':
        text = process_simple_2parts(text, unit)
    text = process_single_digit(text)
    for unit in '十百千万亿':
        text = reduce(text, unit)
    return text


def word2num(s):
    return {
        "天": 0,
        "日": 0,
        "末": 0,
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
    }.get(s)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

