
from .handler import BaseFormHandler, BaseForm, get_entity_from_message, get_entities_from_message
from typing import Text
from starbot.action.db_orm import *


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

    def validate(self):
        form = self.form

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
                products = db_orm_query(Product, thing, thing)
                if not products:
                    self.utter_message("不好意思，我们这里没有{}".format(thing))
                    self.clear_slot('thing')
                    self.clear_slot('count')
                    continue
                for product in cart:
                    if product['thing'] == thing:
                        product['count'] = count
                        break
                else:
                    cart.append({'thing': thing, 'count': count})
            self.set_slot('cart', cart)
            self.utter_message("请问您还需要什么?")
            self.clear_slot('thing')
            self.clear_slot('count')
            return False

        if (n_things, n_counts) not in ((0, 1), (1, 0)):
            self.utter_message("你说啥，我没听清?")
            return False

        # TODO:
        def normalize(x): return x
        if normalize(form.thing) in {'茶叶', '蚊香', '吹风'}:
            if form.count is None:
                form.count = 1

        if form.thing is None:
            thing = self.find_thing_in_history()
            if thing:
                self.form.put_entity('thing', thing)
            else:
                self.utter_message("请问您需要什么?")
                return False

        products = db_orm_query(Product, form.thing, form.thing)
        if not products:
            self.utter_message("不好意思，我们这里没有{}".format(form.thing))
            self.clear_slot('thing')
            self.clear_slot('count')
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
        self.set_slot('cart', cart)
        self.utter_message("请问您还需要什么?")
        self.clear_slot('thing')
        self.clear_slot('count')
        return False

    def commit(self):
        cart = self.get_slot('cart') or []
        things = ''
        for thing in cart:
            things += thing['thing']
        self.utter_message("好的，您要的{}马上为您送过来".format(things))
