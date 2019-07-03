#!/usr/bin/env python
import os
import sys
import time
from rasa.nlu.model import Interpreter
from rasa.model import get_latest_model


def interpret_messages(messages):
    interpreter = Interpreter.load("./models/nlu")

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
        print('intent:', p['intent'])
        for entity in p['entities']:
            print("  {:12}: {}".format(entity['entity'], entity['value']))
    #print(json.dumps(all_result, ensure_ascii=False, indent=2))


if len(sys.argv) == 2:
    interpret_messages([sys.argv[1]])
    sys.exit(0)
else:
    COMMONMSG = [
            '我想订一个小房间',
            '帮我订一间大床房',
            '我叫狄仁杰',
            '明天晚上入住',
            '我今晚入住',
            '帮我订一间今晚的大床房',
            '帮我订一间大床房今晚入住',
            '帮我订一间大床房明晚入住',
            '帮我订一间标间明晚入住',
            '一个大床房今晚的',
            '一个二十五号的标间',
            '大床房一个今晚的',
            '我叫狄仁杰订一间大床房今晚的电话是12345678',
            '我叫如来订一间大床房',
            '我想了解一下价格',
            '感觉不错的样子',
            '对的',
            '我想订单人间',
            '好的',
            'yes',
            '谢谢',
            '瓯海文化互殴',
            '中华人民共和国万岁',
            ]
    #COMMONMSG = json.load(open('test.json'))
    model_file = get_latest_model('../rasa_prj/models/')
    assert model_file
    os.system('rm -rf models && mkdir models && cd models && tar xf {}'.format(model_file))
    interpret_messages(COMMONMSG)
