#!/usr/bin/env python
import sys
from rasa_nlu.model import Interpreter
import json
import time


interpreter = Interpreter.load("./models/current/nlu")


def interpret_messages(messages):
    all_result = []
    for message in messages:
        t0 = time.time()
        text = message
        if isinstance(message, dict):
            text = message['text']
        result = interpreter.parse(text)
        all_result.append({
            'time': time.time() - t0,
            'message': message,
            'prediction': result,
        })
    for example in all_result:
        p = example['prediction']
        print("=="*10)
        print(p['text'])
        for entity in p['entities']:
            print("{}: {}".format(entity['entity'], entity['value']))
    #print(json.dumps(all_result, ensure_ascii=False, indent=2))


if len(sys.argv) == 2:
    interpret_messages([sys.argv[1]])
    sys.exit(0)
else:
    COMMONMSG = [
            '我想订一个小房间',
            '帮我订一间大床房',
            '我叫杰哥',
            '明天晚上入住',
            ]
    #COMMONMSG = json.load(open('test.json'))
    interpret_messages(COMMONMSG)
