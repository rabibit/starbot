from .handler import BaseHandler
from typing import Text, Dict, Any, List
from rasa_sdk.executor import CollectingDispatcher, Tracker
from rasa_sdk.events import SlotSet
import logging


logger = logging.getLogger(__name__)


class SimpleHandler(BaseHandler):
    responses = {
        'account_issues': '没问题，可以直接挂在房帐上面',
        'air_conditioner_problem': '好的，我马上叫人过来帮您看看，很抱歉给您带来不便',
        'any_other?': '不好意思，只有这些了',
        'ask_for_awaking': '您的闹钟已经设定',
        'ask_for_changing_room': '您好，申请换房需要您亲自到前台办理，谢谢',
        'ask_for_charger': '好的，稍后让服务员给您送到房间',
        'ask_for_help': '好的，没问题',
        'ask_for_laundry': '您好，洗衣房在七楼，所有设备全自助',
        'ask_for_more_breakfast_ticket': '不好意思，请稍后，马上给您补上',
        'ask_for_phone_number': '电话号码是123456789',
        'ask_for_price': '8元',
        'ask_for_something_to_eat': '我们这里有饼干和方便面',
        'ask_for_traffic_info': '可以乘坐1路公交或者直接打的',
        'ask_for_wifi_info': 'wifi账号是智慧酒店的拼音首字母，密码是8个8',
        'ask_for_wifi_password': 'wifi账号是智慧酒店的拼音首字母，密码是8个8',
        'ask_how_to_pay': '我们这里可以微信支付，支付宝支付以及刷卡',
        'ask_if_ferry': '我们目前只提供机场的接送服务',
        'ask_price_for_changing_room': '如果换房间的话需要补差价并收取额外的手续费用20元',
        'ask_to_change_thing': '好的，马上安排服务员给您更换',
        'ask_to_clean_room': '好的，马上安排服务员给您打扫',
        'ask_to_open_door': '不好意思，这个已经超出了我的权限，需人工处理',
        'book_room': '订房请拨打123，谢谢',
        'breakfast_ticket_not_found': '您好，早餐券都是统一放在床头的抽屉里面，麻烦再仔细找找，谢谢',
        'buy_or_borrow': '这个是需要单独购买的哟',
        'bye': '再见',
        'can_deliver?': '不好意思，我们餐厅不提供送餐服务',
        'can_i_have_invoice': '可以的',
        'can_order_meal': '可以订餐的',
        'cancel_book_room': '好的，您的预订已经取消',
#        'change_info': '好的，已更正',
        'checkout': '好的，收到',
        'complain': '非常抱歉，给您带来不便',
        'confirm_extend_condition': '目前没有续房',
        'confirm_location': '是成都吧',
        'consultation': '智慧酒店是一家相当棒的酒店',
        'delay_checkin': '好的，已经备注',
        'enha': '呵呵',
        'fetch_it_myself?': '是的，您直接下去拿就是了',
        'greet': '您好，我是小智',
        'hmm': '呵呵',
        'how_far?': '十万八千里',
        'how_much_did_i_spend': '您已经消费八千块',
        'how_much_if_stay_until': '您好，延迟退房需要加收半天的房费',
        'how_to_call': '直接拨号就可以了',
#        'info': '收到',
        'is_breakfast_included': '您好，特价房是不包含早餐的，其它的都包含',
        'is_it_free': '您可以免费使用',
        'is_it_ok': '对的',
        'is_manager_there': '不好意思，经理不在',
        'is_my_cloth_ready': '已经干了，直接去洗衣房取就是了',
        'is_my_room_ready': '可以用了',
        'is_there_any_massage': '酒店附近有一个按摩服务店',
        'is_there_breakfast_now': '不好意思，早餐时间已经过了',
        'is_there_cloth_drier': '有的',
        'is_there_night_snack': '没有',
        'is_there_xxx': '没有',
        'is_there_xxx_around': '附近有一个，出门右转就到了',
        'is_vip_the_same': 'vip可以享受八折优惠',
        'lack_of_thing': '不好意思，服务员马上给您送过来',
        'laundry_request': '拿上来吧',
        'leave_over_something': '好的，帮您看看',
        'network_problem': '不好意思，给您带来不便，马上叫人过来看看',
#        'no': '收到',
#        'not_found': '再仔细找找呢',
#        'number_of_thing': '好的',
#        'ok': '好',
        'order_something': '好的，马上给您送来',
#        'other': '很遗憾，小智无能为力',
        'other_issue_needs_service': '好的，尽快给您处理',
        'query_agreement_price': '标间200，大床房210',
        'query_book_record': '您暂时没有预订记录',
        'query_checkout_time': '我们这里最晚两点退房',
        'query_supper_time': '晚餐是下午五点到晚上九点',
        'repeat_confirm': '对的',
        'room_available': '您好，现在所有房间已经订满，实在不好意思',
        'something_like': '很抱歉，没有的',
        'stay_extension': '可以续住',
#        'this_phone': '好的',
        'tv_problem': '不好意思，给您带来不便，我们马上安排人过来看看',
        'urge': '您好，请耐心等待，我这里帮您催催',
        'wanna_more': '这个需要问我们经理',
        'what_can_you_do': '您需要什么服务呢，小智竭尽全力',
        'when_to_have_breakfast': '早餐是早上七点到九点',
        'when_to_have_lunch': '午餐是十一点到下午两点',
        'where_is_laundry_room': '洗衣房在七楼',
        'where_is_the_wenxiang': '您好，蚊香都是放在床头柜的抽屉里的，请仔细找找',
        'where_is_tv_controller': '电视遥控器是放在写字台的抽屉里的，请仔细找找',
        'where_to_have_breakfast': '早餐在12楼',
    }

    def match(self, tracker: Tracker, domain: Dict[Text, Any]) -> bool:
        return self.get_last_user_intent(tracker) in self.responses

    def process(self,
                dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        intent = self.get_last_user_intent(tracker)
        events = []
        if intent == 'ask_for_price':
            thing = self.get_entity(tracker, 'thing')
            events.append(SlotSet('thing', thing))
            logger.info(f'ask_for_price set slot thing is {thing}')
        dispatcher.utter_message(self.responses.get(intent, 'What?'))
        return events
