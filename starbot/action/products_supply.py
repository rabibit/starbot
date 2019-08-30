#!/usr/bin/env python
# -*- coding: utf-8 -*-

from starbot.action.db_orm import *


products = [
    Product(Name='饼干', Price=12, Keywords="吃的"),
    Product(Name='康师傅方便面', Price=10, Keywords="吃的，方便面"),
    Product(Name='雪碧', Price=4.5, Keywords="喝的"),
    Product(Name='可口可乐', Price=3, Keywords="喝的，可乐，饮料"),
    Product(Name='百事可乐', Price=3, Keywords="喝的，饮料，矿泉水，纯净水"),
    Product(Name='曲奇', Price=3, Keywords="吃的，饼干"),
    Product(Name='怡宝', Price=3, Keywords="喝的，饮料，矿泉水，纯净水"),
    Product(Name='农夫山泉', Price=3, Keywords="喝的，饮料，矿泉水，纯净水"),
    Product(Name='统一方便面', Price=3, Keywords="方便面，吃的")
]

db_orm_add_all(Product, products)
