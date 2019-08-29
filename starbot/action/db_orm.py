#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker


# eng = create_engine('sqlite:///:memory:')
engine = create_engine('sqlite:///seller_bot.db')

Base = declarative_base()


class Product(Base):
    __tablename__ = "products"

    Name = Column(String, primary_key=True)
    Price = Column(Integer)
    # Stock = Column(Integer)
    Keywords = Column(String)
    Delete = Column(Integer, default=0)


Base.metadata.bind = engine
Base.metadata.create_all()

Session = sessionmaker(bind=engine)


def db_orm_add_all(instances: list):
    ses = Session()
    ses.add_all(instances)
    ses.commit()
    ses.close()


# ses.add(Product(Name='怡宝', Price=2, Keywords='饮料,喝的,水'))


# ses.add_all(
#    [Product(Name='花生', Price=12, Eat='yes', Drink='no', Stock=10),
#     Product(Name='饼干', Price=12, Eat='yes', Drink='no', Stock=10),
#     Product(Name='方便面', Price= 3.5, Eat='yes', Drink='no', Stock=23),
#     Product(Name='雪碧', Price=2.3, Eat='no', Drink='yes', Stock=10)]
# )


def db_orm_add(instance):
    ses = Session()
    ses.add(instance)
    ses.commit()
    ses.close()


def db_orm_query_all(orm_obj) -> list:
    ses = Session()
    result = ses.query(orm_obj).all()
    ses.close()
    return result


def db_orm_query(orm_obj, patterns: str) -> list:
    ses = Session()
    result = ses.query(orm_obj).filter(orm_obj.Keywords.like('%' + patterns + '%'), orm_obj.Delete == 0)
    ses.close()
    return result


def db_orm_delete(orm_obj, pattern: str) -> None:
    ses = Session()
    ses.query(orm_obj).filter(orm_obj.Name == pattern).update({"Delete": 1})
    ses.commit()
    ses.close()


if __name__ == '__main__':
    # db_orm_add(Product(Name='西瓜', Price=12, Keywords="吃的"))
    # db_orm_delete(Product, "饼干")
    #rs = db_orm_query_all(Product)
    rs = db_orm_query(Product, "吃的")
    for product in rs:
        print(product.Name, product.Price, product.Keywords, product.Delete)
