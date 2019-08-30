#!/usr/bin/env python
# -*- coding: utf-8 -*-

from starbot.action.db_orm import *


products = [
    Product(Name='饼干', Price=12, Keywords="吃的"),
    Product(Name='方便面', Price=10, Keywords="吃的"),
    Product(Name='雪碧', Price=4.5, Keywords="喝的"),
    Product(Name='可乐', Price=3, Keywords="喝的")
]

db_orm_add_all(Product, products)
