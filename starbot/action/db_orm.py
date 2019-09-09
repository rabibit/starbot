#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker
from operator import or_, and_


# eng = create_engine('sqlite:///:memory:')
engine = create_engine('sqlite:///seller_bot.db')

Base = declarative_base()


class Inform(Base):
    __tablename__ = "informations"

    """
    :param id: id number
    :param name: the name of the object
    :param variety: the type of the object(only support such keys: product, position, service)
    :param price: the price of the product 
    :param keywords: the description of the object
    :param service: the description of the service
    :param position: the description of the position
    :param : phone number
    :param delete: the status of the object
    """

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    variety = Column(String, default="product")
    price = Column(Integer)
    keywords = Column(String)
    service = Column(String)
    position = Column(String)
    contact = Column(String)
    delete = Column(Integer, default=0)


Base.metadata.bind = engine
Base.metadata.create_all()

Session = sessionmaker(bind=engine)


def db_orm_add(instance):
    """
    :param instance: a product e.g: Inform(name='怡宝', variety='product', price=2, keywords='饮料,喝的,水')
    :return: None
    """

    ses = Session()
    ses.add(instance)
    ses.commit()
    ses.close()

    return None


def db_orm_add_all(instances: list):
    """
    :param instances: a set of products e.g: [Iroduct(name='怡宝', variety='product', price=2,keywords='饮料,喝的,水')]
    :return: None
    """

    ses = Session()
    ses.add_all(instances)
    ses.commit()
    ses.close()

    return None


def db_orm_query_all(orm_obj) -> list:
    """
    :param orm_obj: a spicific object base on ORM Base e.g: Inform
    :return: a list of objects
    """

    ses = Session()
    result = ses.query(orm_obj).all()
    ses.close()

    return result


def db_orm_query(orm_obj, patterns: str = '', name: str = '') -> list:
    """
    :param orm_obj: a spicific object base on ORM Base e.g: Inform
    :param patterns: the conditions for the query
    :param name: the name of the object
    :return: a list of objects
    """

    ses = Session()
    if patterns:
        result = ses.query(orm_obj).filter(or_(orm_obj.name == name, and_(orm_obj.keywords.like('%' + patterns + '%'),
                                                                          orm_obj.delete == 0))).limit(5)
    else:
        result = ses.query(orm_obj).filter(orm_obj.name == name).limit(5)
    ses.close()

    return list(result)


def db_orm_delete(orm_obj, pattern: str) -> None:
    """
    :param orm_obj: a spicific object base on ORM Base e.g: Inform
    :param pattern: the  name of the product
    :return: None
    """

    ses = Session()
    ses.query(orm_obj).filter(orm_obj.name == pattern).update({"delete": 1})
    ses.commit()
    ses.close()

    return None


def db_orm_modify(orm_obj, pattern: str, new_informs: dict) -> None:
    """
    :param orm_obj: a spicific object base on ORM Base e.g: Inform
    :param pattern: the  name of the product
    :param new_informs: the informations needed to be modified e.g: {"keywords": "吃的", "price": 23}
    :return: None
    """

    ses = Session()
    ses.query(orm_obj).filter(orm_obj.name == pattern, orm_obj.delete == 0).update(new_informs)
    ses.commit()
    ses.close()

    return None


if __name__ == '__main__':
    # db_orm_add(Inform(name='方便面', price=12, keywords="吃的"))
    # db_orm_add(Inform(name='按摩', variety="service", keywords="按摩服务", service="欢迎来享受极致服务"))
    rs = db_orm_query(Inform, "按摩服务")
    for var in rs:
        if var.variety == "service":
            print(f"{var.service}")
        print(var.id, var.name, var.variety, var.price, var.keywords, var.delete)
