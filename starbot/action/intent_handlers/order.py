from starbot.nlu.timeparser.numberify import numberify
from starbot.action.intent_handlers.handler import (BaseFormHandler,
                                                    BaseForm,
                                                    get_entity_from_message,
                                                    get_entities_from_message)
from typing import Text
from starbot.action.db_orm import *
import re
import logging


logger = logging.getLogger(__name__)


def count_normalized(cnt):
    if len(cnt) == 1 and not numberify(cnt).isdigit():
        return '一' + cnt
    return cnt


def get_number_from_count(count):
    """

    :param count:
    :return:

    examples:
    ---------
    >>> get_number_from_count('一张')
    1
    >>> get_number_from_count('两个')
    2
    >>> get_number_from_count('放肆')
    """
    pattern = re.compile("[0-9]+")
    count = numberify(count)
    match = pattern.search(count)
    if match:
        return int(match.group())
    return None


class SimpleOrderHandler(BaseFormHandler):
    class Form(BaseForm):
        __tag__ = 'order'
        thing: str
        count: str
        number: int
        cart: list
        gpt2prompt: list

    form: Form

    def form_trigger(self, intent: Text):
        return intent == 'order_something'

    def find_thing_in_history(self):
        user_utters = [event for event in self.tracker.events[::-1] if event['event'] == 'user']
        for event in user_utters[1:3]:
            thing = get_entity_from_message(event['parse_data'], 'thing')
            if thing:
                return thing

    def check_count(self, count, thing):
        if count is None:
            self.utter_message(f'请问你需要多少{thing}?')
            return False
        elif len(count) == 2 and count[0] == '几':
            self.utter_message(f'请问你具体需要{count}{thing}?')
            return False
        elif get_number_from_count(count) is None:
            self.utter_message(f'请问你需要多少{thing}?')
            return False
        else:
            return True

    def validate(self, recovering: bool):
        form = self.form

        gpt2out = self.tracker.latest_message.get('gpt2out')
        from_gpt2out = False
        from_modify_info = False
        intent = self.tracker.latest_message.get('intent', {}).get('name')
        modify_info = self.tracker.latest_message.get('modify_info', [])
        print(f'intent: {intent}, modify_info: {modify_info}')
        wrong = None
        right = None

        if intent == 'modify_info':
            for entity in get_entities_from_message(self.tracker.latest_message, 'thing'):
                for info in modify_info:
                    if entity == info['value']:
                        wrong = entity
                    else:
                        right = entity

        print(f'gpt2out: {gpt2out}, wrong: {wrong}, right: {right}')
        if wrong:
            gpt2out = None
            if right:
                form.thing = right
                cart = self.context.get_slot('cart') or []
                new_cart = cart.copy()
                not_found = True
                for product in cart:
                    if product['thing'] == wrong:
                        not_found = False
                        new_cart.remove(product)
                if not_found:
                    self.utter_message(f'不好意思，{wrong}不在您的购物车里面')
                    return False
                else:
                    form.cart = new_cart
                    self.utter_message(f'好的, {wrong}换成{right}')
                from_modify_info = True
            else:
                cart = self.context.get_slot('cart') or []
                new_cart = cart.copy()
                not_found = True
                for product in cart:
                    if product['thing'] == wrong:
                        not_found = False
                        new_cart.remove(product)
                form.cart = new_cart
                if not_found:
                    self.utter_message(f'不好意思，{wrong}不在您的购物车里面')
                    return False
                else:
                    self.utter_message(f'好的, {wrong}不要了，请问您还需要别的什么?')
                return False

        if gpt2out:
            result = db_orm_query(Inform, name=gpt2out)
            for rt in result:
                if rt.variety == 'product':
                    form.thing = gpt2out
                    from_gpt2out = True
                    break
        if recovering:
            if form.count is not None and form.thing is None:
                self.utter_message(f'不好意思, 你刚才说需要{form.count}什么?')
            elif form.thing is not None and form.count is None:
                result = db_orm_query(Inform, name=form.thing)
                for rt in result:
                    if rt.variety == 'product':
                        self.utter_message(f'请问你需要多少{form.thing}?')
                        return False
                result = db_orm_query(Inform, form.thing, form.thing)
                product = False
                products = []
                for rt in result:
                    if rt.variety == 'product':
                        product = True
                        products.append(rt)
                if product:
                    self.prompt_for_chosing(products, form)
                    return False
                else:
                    self.utter_message("不好意思，我们这里没有{}".format(form.thing))
                    form.thing = None
                    form.count = None
                    return False
            elif form.cart:
                things = '，'.join([i["count"] + i['thing'] for i in form.cart])
                if from_gpt2out:
                    self.utter_message(f'您要了{things}，请问你还需要其它吗？', prompt=['used'])
                else:
                    self.utter_message(f'您要了{things}，请问你还需要其它吗？')
            else:
                self.utter_message("你有什么需要吗?")
            return False

        if not gpt2out and self.tracker.latest_message.get('intent', {}).get('name') in {'ok', 'no'}:
            if not form.cart:
                self.utter_message("你还没说你需要啥?")
                return False
            else:
                return True

        things = get_entities_from_message(self.tracker.latest_message, 'thing')
        counts = get_entities_from_message(self.tracker.latest_message, 'count')
        n_things = len(things)
        n_counts = len(counts)
        thing_complement = None
        if n_things == n_counts and n_things >= 1:
            cart = self.context.get_slot('cart') or []
            fuzzy_thing = False
            for thing, count in zip(things, counts):
                if len(thing) == 1 and thing != '水':
                    self.skip()
                    return
                count = count_normalized(count)
                result = db_orm_query(Inform, name=thing)
                product = False
                for rt in result:
                    if rt.variety == 'product':
                        product = True
                if not product:
                    result = db_orm_query(Inform, thing, thing)
                    product = False
                    products = []
                    for rt in result:
                        if rt.variety == 'product':
                            product = True
                            products.append(rt)
                    if product:
                        if len(products) == 1:
                            thing_complement = thing + '(' + products[0].name + ')'
                        else:
                            self.prompt_for_chosing(products, form)
                            form.count = count
                            if cart:
                                self.context.set_slot('cart', cart)
                            return False
                    else:
                        self.utter_message("不好意思，我们这里没有{}".format(thing))
                        form.thing = None
                        form.count = None
                        return False
                if not self.check_count(count, thing):
                    fuzzy_thing = thing
                    continue
                for product in cart:
                    if product['thing'] == thing:
                        product['count'] = count
                        break
                else:
                    cart.append({'thing': thing_complement if thing_complement else thing, 'count': count})
                self.utter_message(f'{count}{thing}')
            self.context.set_slot('cart', cart)
            if fuzzy_thing:
                form.thing = fuzzy_thing
            else:
                if from_gpt2out:
                    self.utter_message("请问您还需要什么?", prompt=['used'])
                else:
                    self.utter_message("请问您还需要什么?")
                form.thing = None
            form.count = None
            return False

        if (n_things, n_counts) not in ((0, 1), (1, 0)) and not from_gpt2out and not from_modify_info:
            self.skip()
            # self.skip_if_intended()
            # self.utter_message("你说啥，我没听清?")
            return

        # TODO:
        def normalize(x): return x
        if normalize(form.thing) in {'茶叶', '蚊香', '吹风'}:
            if form.count is None:
                form.count = '一'

        if form.thing is None:
            form.thing = self.find_thing_in_history()

        if form.thing is None:
            self.utter_message("请问您需要什么?")
            return False

        result = db_orm_query(Inform, name=form.thing)
        product = False
        for rt in result:
            if rt.variety == 'product':
                product = True
        if not product:
            result = db_orm_query(Inform, form.thing, form.thing)
            product = False
            products = []
            for rt in result:
                if rt.variety == 'product':
                    product = True
                    products.append(rt)
            if product:
                if len(products) == 1:
                    thing_complement = form.thing + '(' + products[0].name + ')'
                else:
                    self.prompt_for_chosing(products, form)
                    return False
            else:
                self.utter_message("不好意思，我们这里没有{}".format(form.thing))
                form.thing = None
                form.count = None
                return False

        if not self.check_count(form.count, form.thing):
            return False

        cart = self.context.get_slot('cart') or []
        count = count_normalized(form.count)
        for product in cart:
            if product['thing'] == form.thing:
                product['count'] = count
                break
        else:
            cart.append({'thing': thing_complement if thing_complement else form.thing, 'count': count})
        form.cart = cart
        if from_gpt2out:
            self.utter_message(f"{form.count}{form.thing}，请问您还需要什么?", prompt=['used'])
        else:
            self.utter_message(f"{form.count}{form.thing}，请问您还需要什么?")
        form.thing = None
        form.count = None
        return False

    def commit(self):
        cart = self.form.cart or []
        things = ''
        if cart:
            cart = [item['thing'] for item in cart]
            last = cart[-1]
            start = '，'.join(cart[:-1])
            if start:
                things = '和'.join([start, last])
            else:
                things = last
        self.utter_message("好的，您要的{}马上为您送过来".format(things))

    def cancel(self, force: bool):
        logger.info(f"canceling order force={force} cart={self.form.cart}")
        if not force and self.form.cart:
            self.utter_message("不想要了你可以说，返回")
            self.abort()
        else:
            super(SimpleOrderHandler, self).cancel(force)

    def prompt_for_chosing(self, products, form):
        self.utter_message("我们这里有：")
        for food in products:
            self.utter_message("{}".format(food.name), prompt=[food.name for food in products])
        self.utter_message("请问您要哪一种")
        form.thing = None
        #form.count = None
