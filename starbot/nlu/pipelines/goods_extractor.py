import re

from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.training_data import Message
from starbot.action.db_orm import *

reg_str = r"""(百事|可口)(可乐)?
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
|勇闯天涯
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
|奔驰
|宝马
|电风扇
|空调
|统一方便面
|康师傅方便面
|康师傅
|纯果乐
|(百威|青岛)啤酒
|纯生
"""

result = db_orm_query_all(Inform)
for rt in result:
    if rt.variety == 'product':
        reg_str = reg_str + '\n|' + rt.name

pattern = re.compile(reg_str, re.X | re.I)


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
            entities.sort(key=lambda x: x['start'])
            message.set("entities", entities, add_to_output=True)
            modify_infos = message.get('modify_info') or []
            modify_infos = merge_entities_goods(modify_infos, brands, keep=True)
            modify_infos.sort(key=lambda x: x['start'])
            message.set("modify_info", modify_infos, add_to_output=True)


def merge_entities_goods(entities: list, goods: list, keep: bool=False) -> list:
    """

    :param entities: got from network
    :param goods: got form  regex
    :param keep: if the entity name will be changed
    :return: entities modified by goods
    >>> merge_entities_goods([], [])
    []

    >>> merge_entities_goods([], [{'start': 4, 'end': 6, 'entity': 'goods', 'value': '可乐'}])
    [{'start': 4, 'end': 6, 'entity': 'thing', 'value': '可乐'}]

    >>> merge_entities_goods([], [{'start': 4, 'end': 6, 'entity': 'goods', 'value': '可乐'}], keep=True)
    []

    >>> merge_entities_goods([{'start': 0, 'end': 2, 'entity': 'goods', 'value': '百事'},
    ... {'start': 2, 'end': 4, 'entity': 'goods', 'value': '可乐'},
    ... {'start': 4, 'end': 6, 'entity': 'goods', 'value': '好喝'}],
    ... [{'start': 1, 'end': 5, 'entity': 'goods', 'value': '事可乐好'}])
    [{'start': 1, 'end': 5, 'entity': 'thing', 'value': '事可乐好'}]

    >>> merge_entities_goods([{'start': 2, 'end': 5, 'entity': 'thing', 'value': '方便面'}],
    ... [{'start': 0, 'end': 5, 'entity': 'goods', 'value': '统一方便面'}])
    [{'start': 0, 'end': 5, 'entity': 'thing', 'value': '统一方便面'}]

    >>> merge_entities_goods([{'start': 2, 'end': 5, 'entity': 'wrong', 'value': '方便面'}],
    ... [{'start': 0, 'end': 5, 'entity': 'goods', 'value': '统一方便面'}], keep=True)
    [{'start': 0, 'end': 5, 'entity': 'wrong', 'value': '统一方便面'}]

    >>> merge_entities_goods([{'start': 0, 'end': 4, 'entity': 'thing', 'value': '康师傅'}],
    ... [{'start': 0, 'end': 6, 'entity': 'goods', 'value': '康师傅方便面'}])
    [{'start': 0, 'end': 6, 'entity': 'thing', 'value': '康师傅方便面'}]
    """

    result = entities.copy()
    if keep:
        for good in goods:
            entity_tmp = None
            merge = False
            for entity in entities:
                if good['start'] <= entity['start'] <= good['end'] or good['start'] <= entity['end'] <= good['end']:
                    entity_tmp = entity
                    result.remove(entity)
                    merge = True
            if merge:
                result.append({
                    "start": good['start'],
                    "end": good['end'],
                    "entity": entity_tmp['entity'],
                    "value": good['value']
                })
    else:
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
