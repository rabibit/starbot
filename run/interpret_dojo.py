#!/usr/bin/env python
import sys
from rasa_nlu.model import Interpreter
import json


interpreter = Interpreter.load("./models/current/nlu")


def interpret_all(messages):
    for message in messages:
        result = interpreter.parse(message)
        print("================= {} ====================".format(message))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if len(sys.argv) == 2:
    interpret_all([sys.argv[1]])
    sys.exit(0)
else:
    COMMONMSG = [
            'hello',
            "哈罗",
            "你好",
            "好吧",
            ]
    interpret_all(COMMONMSG)
