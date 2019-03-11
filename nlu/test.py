#!/usr/bin/env python
import sys
from rasa_nlu.model import Interpreter
import json
interpreter = Interpreter.load("./models/current/nlu")

def interpret_all(messages):
    for message in messages:
        result = interpreter.parse(message)
        del result['intent_ranking']
        print("================= {} ====================".format(message))
        print(json.dumps(result, ensure_ascii=False, indent=2))

if len(sys.argv) == 2:
    interpret_all([sys.argv[1]])
    sys.exit(0)

IPREQS = [
        "我要一个IP",
        "我要一个临时IP",
        "帮我申请个临时IP",
        "帮我申请个固定IP地址",
        "帮我申请个永久IP",
        ]


COMMONMSG = [
        'hello',
        "哈罗",
        "你好",
        "好吧",
        "可以",
        "太好了",
        "不要",
        "我不要这个",
        "OK",
        "不要这样",
        "放首歌",
        "玩音乐",
        "放音乐",
        "我要一个固定IP",
        ]



PHONEREQS = [
        "小明的电话是多少",
        "帮我查查小明的邮箱",
        "帮我看看大鹏的手机是",
        ]
interpret_all(PHONEREQS)
