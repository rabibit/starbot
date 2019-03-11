# coding: utf8
from rasa_core_sdk import Action
from rasa_core_sdk.events import BotUttered
import random
import json


class ActionLookupInfo(Action):
    contacts = json.load(open("contacts.json"))

    def name(self):
        # type: () -> Text
        return "lookup_info"

    def run(self, dispatcher, tracker, domain):
        # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict[Text, Any]]

        def format(info, item):
            value = info['name'] + '\n'
            if item in ("电话", "手机", "号码"):
                value += " 手机: %s" % info["cellphone"]
                if info["ext"].strip():
                    value += " 分机: %s" % info["ext"]
            elif item == "邮箱":
                value += " 邮箱: %s" % info["email"]
            else:
                value += " 手机: %s" % info["cellphone"]
                if info["ext"].strip():
                    value += " 分机: %s" % info["ext"]
                value += " 邮箱: %s" % info["email"]
            return value

        name = tracker.get_slot('lookup_name')
        item = tracker.get_slot('lookup_item')
        info = self.contacts.get(name)
        if not info:
            dispatcher.utter_message("没有找到{}的信息".format(name))
            return []
        if len(info) > 1:
            dispatcher.utter_message("公司一共有{}个{}".format(len(info), name))
        msg = ""
        for i in info:
            msg += format(i, item)
        dispatcher.utter_message(msg)
        return []


class ActionPlayMusic(Action):
    def name(self):
        # type: () -> Text
        return "play_music"

    def run(self, dispatcher, tracker, domain):
        # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict[Text, Any]]
        def random_music(artist):
            if artist:
                mlst = {
                    "周杰伦": ["双截棍", "发如雪", "烟花易冷"],
                    "陈奕迅": ["十年", "浮夸"],
                }.get(artist)
                return random.choice(mlst) if mlst else None
            else:
                return random.choice(["卡路里", "小苹果"])

        name = tracker.get_slot('music_name')
        artist = tracker.get_slot('music_artist')
        if not name:
            if artist:
                name = random_music(artist)
                if not name:
                    dispatcher.utter_message("对不起，没有找到{}的歌曲".format(artist))
                    artist = "周杰伦"
                    name = random_music(artist)
            else:
                name = random_music(None)
        if artist:
            msg = "正在为您播放{}的《{}》".format(artist, name)
        else:
            msg = "正在为您播放《{}》".format(name)
        dispatcher.utter_message(msg)
        return []
