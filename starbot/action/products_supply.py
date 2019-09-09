#!/usr/bin/env python
# -*- coding: utf-8 -*-

from starbot.action.db_orm import *


Informs = [
    Inform(name='饼干', price=12, keywords="吃的"),
    Inform(name='康师傅方便面', price=10, keywords="吃的，方便面"),
    Inform(name='雪碧', price=4.5, keywords="喝的"),
    Inform(name='可口可乐', price=3, keywords="喝的，可乐，饮料"),
    Inform(name='百事可乐', price=3, keywords="喝的，饮料，矿泉水，纯净水"),
    Inform(name='曲奇', price=3, keywords="吃的，饼干"),
    Inform(name='怡宝', price=3, keywords="喝的，饮料，矿泉水，纯净水"),
    Inform(name='农夫山泉', price=3, keywords="喝的，饮料，矿泉水，纯净水"),
    Inform(name='统一方便面', price=3, keywords="方便面，吃的"),
    Inform(name='按摩', variety="service", keywords="按摩服务", service="我们将为您提供极致的按摩服务"),
    Inform(name='红旗连锁', variety="position", keywords="超市", position="就在出门右转50米左右的地方"),
    Inform(name='凯乐迪', variety="position", keywords="KTV", position="就在出门左转150米左右的地方", contact="123456789")
]

db_orm_add_all(Informs)
