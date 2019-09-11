from starbot.nlu.timeparser.numberify import numberify
from .handler import BaseFormHandler, BaseForm, get_entity_from_message, get_entities_from_message
from typing import Text
from starbot.action.db_orm import *
import re


class SimpleOrderHandler(BaseFormHandler):
    class Form(BaseForm):
        __tag__ = 'order'
        thing: str
        count: int
        number: int
        cart: list

    form: Form

    def form_trigger(self, intent: Text):
        return intent == 'order_something'

    def find_thing_in_history(self):
        user_utters = [event for event in self.tracker.events[::-1] if event['event'] == 'user']
        for event in user_utters[1:3]:
            thing = get_entity_from_message(event['parse_data'], 'thing')
            if thing:
                return thing

    def validate(self, recovering: bool):
        form = self.form

        if recovering:
            if form.count is not None and form.thing is None:
                self.utter_message(f'不好意思, 你刚才说需要{form.count}什么?')
            elif form.thing is not None and form.count is None:
                self.utter_message(f'请问你需要多少{form.thing}?')
            elif form.cart:
                things = '，'.join([i["count"] + i['thing'] for i in form.cart])
                self.utter_message(f'您要了{things}，请问你还需要其它吗？')
            else:
                self.utter_message("你有什么需要吗?")
            return False

        if self.tracker.latest_message.get('intent', {}).get('name') == 'ok':
            if not form.cart:
                self.utter_message("你还没说你需要啥?")
                return False
            else:
                return True

        things = get_entities_from_message(self.tracker.latest_message, 'thing')
        counts = get_entities_from_message(self.tracker.latest_message, 'count')
        n_things = len(things)
        n_counts = len(counts)
        if n_things == n_counts and n_things >= 1:
            cart = self.get_slot('cart') or []
            for thing, count in zip(things, counts):
                result = db_orm_query(Inform, thing, thing)
                for product in result:
                    if product.variety == "product":
                        break
                else:
                    self.utter_message("不好意思，我们这里没有{}".format(thing))
                    form.thing = None
                    form.count = None
                    continue
                for product in cart:
                    if product['thing'] == thing:
                        product['count'] = count
                        break
                else:
                    cart.append({'thing': thing, 'count': count})
            self.set_slot('cart', cart)
            self.utter_message("请问您还需要什么?")
            form.thing = None
            form.count = None
            return False

        if (n_things, n_counts) not in ((0, 1), (1, 0)):
            self.skip_if_intended()
            self.utter_message("你说啥，我没听清?")
            return False

        # TODO:
        def normalize(x): return x
        if normalize(form.thing) in {'茶叶', '蚊香', '吹风'}:
            if form.count is None:
                form.count = 1

        if form.thing is None:
            form.thing = self.find_thing_in_history()

        if form.thing is None:
            self.utter_message("请问您需要什么?")
            return False

        result = db_orm_query(Inform, form.thing, form.thing)
        for product in result:
            if product.variety == "product":
                break
        else:
            self.utter_message("不好意思，我们这里没有{}".format(form.thing))
            form.thing = None
            form.count = None
            return False

        if form.count is None:
            self.utter_message("请问您需要多少{}?".format(form.thing))
            return False

        cart = self.get_slot('cart') or []
        for product in cart:
            if product['thing'] == form.thing:
                product['count'] = form.count
                break
        else:
            cart.append({'thing': form.thing, 'count': form.count})
        form.cart = cart
        self.utter_message("请问您还需要什么?")
        form.thing = None
        form.count = None
        return False

    def commit(self):
        #cart = self.get_slot('cart') or []
        cart = self.form.cart or []
        things = ''
        pattern = re.compile("[0-9]+")
        for thing in cart:
            count = numberify(thing['count'])
            count = pattern.search(count).group()
            print(f"count is {count}")
            things += thing['thing']
        self.utter_message("好的，您要的{}马上为您送过来".format(things))

    def cancel(self, force: bool):
        # TODO: check whether in progress if force = False
        return super(SimpleOrderHandler, self).cancel(force)
