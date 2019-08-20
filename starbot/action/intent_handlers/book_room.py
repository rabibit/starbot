
from .handler import BaseFormHandler, BaseForm
from typing import Text


class BookRoomHandler(BaseFormHandler):
    class Form(BaseForm):
        __tag__ = 'book_room'
        room_type: str
        count: int
        number: int
        person: str
        date: str

    form: Form

    def form_trigger(self, intent: Text):
        return intent == 'book_room'

    def validate(self):
        form = self.form

        if form.room_type is None:
            self.utter_message("请问您要订什么房间？")
            return False

        if form.count is None:
            self.utter_message("请问您需要订几间?")
            return False

        if form.person is None:
            self.utter_message("请问怎么称呼您？")

        if form.date is None:
            self.utter_message("请问您什么时候入住？")

        if form.number is None:
            self.utter_message("请问您的联系方式是？")
            return False
        return True

    def commit(self):
        self.utter_message("好的，您的订房信息为:姓名[{}]，房间类型[{}]，房间数量[{}]，入住时间[{}]，联系电话[{}]"
                                 .format(self.form.person, self.form.room_type, self.form.count, self.form.date,
                                         self.form.number))

