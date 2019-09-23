import re

from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.training_data import Message


pattern = re.compile(r"""(百事|可口)(可乐)?
|可乐
|农夫山泉
|怡宝
|矿泉水
|雀巢(咖啡)?
|咖啡
|康师傅
|巴黎水
|椰汁
|豆奶
|花生奶
|王老吉
|橙汁
|汇源
|百岁山
|六个核桃
|依云
|加多宝
|健力宝
|娃哈哈
|北冰洋
|酸梅汤
|雪碧
|芬达
|七喜
|美年达
|农夫果园
|果粒橙
|纯果乐
|(百威|青岛)啤酒
|纯生
""", re.X | re.I)


class GoodsExtractor(EntityExtractor):
    provides = ['entities']

    def process(self, message: Message, **kwargs) -> None:
        """

        :param message:
        :param kwargs:
        :return:

        >>> msg = Message('百事可乐吧')
        >>> GoodsExtractor().process(msg)
        >>> msg.get('entities')
        [{'start': 0, 'end': 4, 'entity': 'goods', 'value': '百事可乐', 'extractor': 'GoodsExtractor'}]

        >>> msg = Message('百事吧')
        >>> GoodsExtractor().process(msg)
        >>> msg.get('entities')
        [{'start': 0, 'end': 2, 'entity': 'goods', 'value': '百事', 'extractor': 'GoodsExtractor'}]

        """
        brands = []
        for m in pattern.finditer(message.text):
            brands.append({
                "start": m.span()[0],
                "end": m.span()[1],
                "entity": "goods",
                "value": m.group(),
                "extractor": "GoodsExtractor"
            })
        if brands:
            entities = message.get('entities') or []
            entities += brands
            message.set("entities", entities, add_to_output=True)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
