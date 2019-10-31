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
|统一方便面
|康师傅方便面
|康师傅
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
        [{'start': 0, 'end': 4, 'entity': 'thing', 'value': '百事可乐'}]

        >>> msg = Message('百事吧')
        >>> GoodsExtractor().process(msg)
        >>> msg.get('entities')
        [{'start': 0, 'end': 2, 'entity': 'thing', 'value': '百事'}]

        >>> msg = Message('统一方便面')
        >>> GoodsExtractor().process(msg)
        >>> msg.get('entities')
        [{'start': 0, 'end': 5, 'entity': 'thing', 'value': '统一方便面'}]

        >>> msg = Message('康师傅方便面')
        >>> GoodsExtractor().process(msg)
        >>> msg.get('entities')
        [{'start': 0, 'end': 6, 'entity': 'thing', 'value': '康师傅方便面'}]
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
            entities = merge_entities_goods(entities, brands)
            message.set("entities", entities, add_to_output=True)


def merge_entities_goods(entities: list, goods: list) -> list:
    """

    :param entities: got from network
    :param goods: got form  regex
    :return: entities modified by goods
    >>> merge_entities_goods([], [])
    []

    >>> merge_entities_goods([], [{'start': 4, 'end': 6, 'entity': 'goods', 'value': '可乐'}])
    [{'start': 4, 'end': 6, 'entity': 'thing', 'value': '可乐'}]

    >>> merge_entities_goods([{'start': 0, 'end': 2, 'entity': 'goods', 'value': '百事'},
    ... {'start': 2, 'end': 4, 'entity': 'goods', 'value': '可乐'},
    ... {'start': 4, 'end': 6, 'entity': 'goods', 'value': '好喝'}],
    ... [{'start': 1, 'end': 5, 'entity': 'goods', 'value': '事可乐好'}])
    [{'start': 1, 'end': 5, 'entity': 'thing', 'value': '事可乐好'}]

    >>> merge_entities_goods([{'start': 2, 'end': 5, 'entity': 'thing', 'value': '方便面'}],
    ... [{'start': 0, 'end': 5, 'entity': 'goods', 'value': '统一方便面'}])
    [{'start': 0, 'end': 5, 'entity': 'thing', 'value': '统一方便面'}]

    >>> merge_entities_goods([{'start': 0, 'end': 4, 'entity': 'thing', 'value': '康师傅'}],
    ... [{'start': 0, 'end': 6, 'entity': 'goods', 'value': '康师傅方便面'}])
    [{'start': 0, 'end': 6, 'entity': 'thing', 'value': '康师傅方便面'}]
    """

    result = entities.copy()
    for good in goods:
        for entity in entities:
            if good['start'] <= entity['start'] <= good['end'] or good['start'] <= entity['end'] <= good['end']:
                result.remove(entity)
        result.append({
            "start": good['start'],
            "end": good['end'],
            "entity": 'thing',
            "value": good['value']
        })
    return result


if __name__ == '__main__':
    import doctest
    doctest.testmod()
