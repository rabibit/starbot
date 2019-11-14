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



def patch_rasa_for_tf2():
    import tensorflow as tf
    import sys
    tf.logging = tf.compat.v1.logging
    tf.ConfigProto = None

    sys.modules['rasa.core.policies.embedding_policy'] = type(sys)("embedding_policy")
    sys.modules['rasa.core.policies.embedding_policy'].EmbeddingPolicy = None
    sys.modules['rasa.core.policies.keras_policy'] = type(sys)("embedding_policy")
    sys.modules['rasa.core.policies.keras_policy'].KerasPolicy = None

    class EmbeddingIntentClassifier:
        name = "EmbeddingIntentClassifier"

    sys.modules['rasa.nlu.classifiers.embedding_intent_classifier'] = type(sys)("")
    sys.modules['rasa.nlu.classifiers.embedding_intent_classifier'].EmbeddingIntentClassifier = EmbeddingIntentClassifier


def patch_rasa():
    from typing import Text, Optional
    from rasa import model
    from rasa.model import persist_fingerprint, Fingerprint

    def create_package_rasa(
            training_directory: Text,
            output_filename: Text,
            fingerprint: Optional[Fingerprint] = None,
    ) -> Text:
        """Creates a zipped Rasa model from trained model files.

        Args:
            training_directory: Path to the directory which contains the trained
                                model files.
            output_filename: Name of the zipped model file to be created.
            fingerprint: A unique fingerprint to identify the model version.

        Returns:
            Path to zipped model.

        """
        import tarfile

        if fingerprint:
            persist_fingerprint(training_directory, fingerprint)

        output_directory = os.path.dirname(output_filename)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        print(f'output: {output_filename}')

        with tarfile.open(output_filename, "w") as tar:
            for elem in os.scandir(training_directory):
                tar.add(elem.path, arcname=elem.name)
                print(f'add {elem.path}')

        shutil.rmtree(training_directory)
        return output_filename

    model.create_package_rasa = create_package_rasa


if len(sys.argv) == 2:
    patch_rasa()
    patch_rasa_for_tf2()
    interpret_messages([sys.argv[1]])
    sys.exit(0)
else:
    patch_rasa()
    patch_rasa_for_tf2()
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
