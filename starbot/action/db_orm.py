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

    Id = Column(Integer, primary_key=True, autoincrement=True)
    Name = Column(String)
    Price = Column(Integer)
    # Stock = Column(Integer)
    Keywords = Column(String)
    Delete = Column(Integer, default=0)


Base.metadata.bind = engine
Base.metadata.create_all()

Session = sessionmaker(bind=engine)


def db_orm_add(orm_obj, instance):
    """
    :param orm_obj: a spicific object base on ORM Base e.g: Product
    :param instance: a product e.g: Product(Name='怡宝', Price=2, Keywords='饮料,喝的,水')
    :return: None
    """

    ses = Session()
    ses.add(instance)
    ses.commit()
    ses.close()

    return None


def db_orm_add_all(orm_obj, instances: list):
    """
    :param orm_obj: a spicific object base on ORM Base e.g: Product
    :param instances: a set of products e.g: [Product(Name='怡宝', Price=2, Keywords='饮料,喝的,水')]
    :return: None
    """

    ses = Session()
    ses.add_all(instances)
    ses.commit()
    ses.close()

    return None


# ses.add(Product(Name='怡宝', Price=2, Keywords='饮料,喝的,水'))


# ses.add_all(
#    [Product(Name='花生', Price=12, Eat='yes', Drink='no', Stock=10),
#     Product(Name='饼干', Price=12, Eat='yes', Drink='no', Stock=10),
#     Product(Name='方便面', Price= 3.5, Eat='yes', Drink='no', Stock=23),
#     Product(Name='雪碧', Price=2.3, Eat='no', Drink='yes', Stock=10)]
# )


def db_orm_query_all(orm_obj) -> list:
    """
    :param orm_obj: a spicific object base on ORM Base e.g: Product
    :return: a list of objects
    """

    ses = Session()
    result = ses.query(orm_obj).all()
    ses.close()

    return result


def db_orm_query(orm_obj, patterns: str) -> list:
    """
    :param orm_obj: a spicific object base on ORM Base e.g: Product
    :param patterns: the conditions for the query
    :return: a list of objects
    """

    ses = Session()
    result = ses.query(orm_obj).filter(orm_obj.Keywords.like('%' + patterns + '%'), orm_obj.Delete == 0)
    ses.close()

    return result


def db_orm_delete(orm_obj, pattern: str) -> None:
    """
    :param orm_obj: a spicific object base on ORM Base e.g: Product
    :param pattern: the  name of the product
    :return: None
    """

    ses = Session()
    ses.query(orm_obj).filter(orm_obj.Name == pattern).update({"Delete": 1})
    ses.commit()
    ses.close()

    return None


def db_orm_modify(orm_obj, pattern: str, keywords: str) -> None:
    """
    :param orm_obj: a spicific object base on ORM Base e.g: Product
    :param pattern: the  name of the product
    :param keywords: the discription of the product
    :return: None
    """

    ses = Session()
    ses.query(orm_obj).filter(orm_obj.Name == pattern, orm_obj.Delete == 0).update({"Keywords": keywords})
    ses.commit()
    ses.close()

    return None


if __name__ == '__main__':
    db_orm_add(Product, Product(Name='方便面', Price=12, Keywords="吃的"))
    db_orm_add_all(Product, [Product(Name='饼干', Price=12, Keywords="吃的")])
    db_orm_delete(Product, 'ddd')
    rs = db_orm_query_all(Product)
    for product in rs:
        print(product.Id, product.Name, product.Price, product.Keywords, product.Delete)
