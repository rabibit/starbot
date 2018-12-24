#!/usr/bin/env python
import sys
from rasa_nlu.model import Interpreter
import json
interpreter = Interpreter.load("./models/current/nlu")

for message in ['hello', "哈罗", "你好", "好吧", "可以", "太好了", "不要", "我不要这个", "OK", "不要这样"]:
  result = interpreter.parse(message)
  print("================= {} ====================".format(message))
  print(json.dumps(result, indent=2))
