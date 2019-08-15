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
            '打卡机表示朕乏了',
            '举报，豆腐渣工程',
            '这是碰瓷啊',
            '会不会拓奇那边用了',
            '有没有人想换花生油的',
            '能不能帮我倒水',
            '能不能帮我打水',
            '能不能帮我办卡',
            '能不能帮我办事',
            '能不能帮我讲题',
            '能不能帮我讲一下作业',
            '不能安排统一查询么',
            '我想订一个小猪',
            '我想订一只烤鸭',
            ]
    COMMONMSG = [
                '可以挂账吗',
                '你好，这个空调开不了啊',
                '你好，还有其他需要吗',
                '你好，明天早上叫我起床好么',
                '你好，这个房间太闷了，我想换一个',
                '你好，有华为的充电器吗',
                '你好，请问洗衣服在哪儿',
                '你好，你好像没给我在餐券',
                '你好，请问餐厅电话是多少',
                '你好，单间多少钱',
                '你好，有什么吃的',
                '你好，这儿去火车南站怎么走',
                '你好，可以用微信吗',
                '你好，你们可以接送是吧',
                '你好，换房间是多少钱',
                '你好，这个床单很久没换了，帮我换一下吧',
                '你好，叫服务员来帮我打扫一下房间吧',
                '你好，我那个房卡锁里面了，帮我开一下们吧1106',
                '你好，那个没看到早餐券呢',
                '你好，是要买是吧',
                '你好，可以送到房间么',
                '可以开发票吗',
                '可以订餐吗',
                '我在美团定了一间房，我暂时来不了了，可以帮我退了吗',
                '北京是吧',
                '怎么还没来呀',
                '809续住了吗',
                '我订了今晚单间，可能晚一点过来',
                '我自己下去拿是吗',
                '这儿过去远吗',
                '我一共消费了多少了',
                '如果4点退房多少钱',
                '外线怎么打',
                '含早餐吧',
                '是免费的吗',
                '你们经理在吗',
                '我的衣服洗好了么，干了吗',
                '我的房间开好了吗',
                '这儿有按摩是吧',
                '现在可以吃早餐了吗',
                '有烘干机吗',
                '有宵夜吗',
                '有没有矿泉水',
                '附近有超市吗',
                '会员也是一样吗',
                '茶叶没有了',
                '我要洗衣服',
                '我有一个钥匙落房间里了',
                '电脑上不了网',
                '没看到哦',
                '5个',
                '要一包玉溪',
                '你好，马桶堵了，过来看一下好吧',
                '星网锐捷的协议价是多少',
                '帮我查一下我订了几间房',
                '帮我看一下订到哪天',
                '晚餐几点',
                '没有了是吧',
                '单间还有没',
                '帮我续到明天',
                '电视机怎么打不开',
                '能不能快点',
                '发票可以多开吗',
                '请问早餐几点',
                '午餐几点',
                '洗衣房在哪儿',
                '蚊香在哪儿',
                '电视机遥控器在哪儿',
                '早餐在几楼',
                '你这个wifi密码是啥呢',
                '我想问一下wifi怎么连',
                '无线网有吗',
                '你好，我是刚刚入住的，请问你们这儿的无线怎么连呢',
                '额 你好 额 我 我成都的',
                '额 你好 额 我 我成都的 叫张全蛋那',
                '额 你好 额 我 我成都的 叫张全蛋那 经常过来住的',
                '你刚才打电话过来的吗',
                '是你刚才打过来的吗',
                '你找我有事吗',
                '我打电话到订票处了',
            ]
    #COMMONMSG = json.load(open('test.json'))
    model_file = get_latest_model(os.path.abspath('rasa_prj/models/'))
    print(f'Using model file: {model_file}')
    assert model_file
    os.system('rm -rf models && mkdir models && cd models && tar xf {}'.format(model_file))
    interpret_messages(COMMONMSG)
