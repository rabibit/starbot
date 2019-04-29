#!/usr/bin/env python
import sys
from rasa_nlu.model import Interpreter
import json


interpreter = Interpreter.load("./models/current/nlu")


def interpret_all(messages):
    for message in messages:
        import time
        t0 = time.time()
        result = interpreter.parse(message)
        print("================= {} time:{} ====================".format(message, time.time() - t0))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if len(sys.argv) == 2:
    interpret_all([sys.argv[1]])
    sys.exit(0)
else:
    COMMONMSG = [
            '我想订一个小房间',
            '我想订一间大床房',
            '我叫杰哥',
            ]
    interpret_all(COMMONMSG)
