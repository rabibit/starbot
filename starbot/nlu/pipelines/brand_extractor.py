import re

from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.training_data import Message

pattern = re.compile(r"""iPhone
|iPad
|Android
|安卓
|Type C
|苹果
|华为
|荣耀
|小米
|一加
|vivo
|魅族
|努比亚
|联想
|诺基亚
|OPPO
|三星
|ZTE
|中兴
|摩托罗拉
|海信
|锤子
|魅蓝
|国美
|索尼
|HTC
|酷比
|夏普
|360
|金立
|SUGAR
|康佳
|华硕
|黑莓
|小辣椒
|飞利浦
|酷派
|美图
|ivvi
|Gigaset
|邦华
|imoo
|长虹
|TCL
|乐视
|蓝魔
|LG
|ZUK
|守护宝
|iuni
|青葱
|富可视
|微软
|THL
|青橙
|百分之百
|神舟
|卓普
|百事
|索尼爱立信
|索爱
|大可乐
|天语
|龙酷
|亚马逊
|谷歌
|纽曼
|现代
|影驰
|云狐
|爱国者
|北斗
|经纬
|盛大
|优派
|戴尔
|宏碁
|海尔
|HKC
|技嘉
|惠普
|NEC
|富士通
|Palm
|夏新
|阿尔卡特
|东芝
|京瓷
|卡西欧
|英华达
|基伍
|万利达
|HIKe
|波导
|朵唯
|迪士尼
|华晶科技
|果冻
|泰丰
|欧恩
|多普达
|读者
|首派
|中恒
|摩西
|139易
|泛泰
|nibiru
|宇达
|锋达通
|英华通
|i-mate
|齐乐
|优米
|七喜
|凡尔纳
|创维
|E人E本
|一人一本
|小蜜蜂
|小宇宙
|格力
|佳域
|8848
|美晨
|国虹
|路虎
|博迪
|奥克斯
|尼凯恩
|PPTV
""", re.X)


class BrandExtractor(EntityExtractor):
    provides = ['entities']

    def process(self, message: Message, **kwargs) -> None:
        """

        :param message:
        :param kwargs:
        :return:

        >>> msg = Message('苹果的华为的')
        >>> BrandExtractor().process(msg)
        >>> msg.get('entities')
        [{'start': 0, 'end': 2, 'entity': 'brand', 'value': '苹果', 'extractor': 'BrandExtractor'}, {'start': 3, 'end': 5, 'entity': 'brand', 'value': '华为', 'extractor': 'BrandExtractor'}]
        """
        brands = []
        for m in pattern.finditer(message.text):
            brands.append({
                "start": m.span()[0],
                "end": m.span()[1],
                "entity": "brand",
                "value": m.group(),
                "extractor": "BrandExtractor"
            })
        if brands:
            entities = message.get('entities') or []
            entities += brands
            message.set("entities", entities, add_to_output=True)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
