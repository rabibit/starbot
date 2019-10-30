from starbot.action.intent_handlers.handler import BaseHandler


class GreetHandler(BaseHandler):
    def match(self) -> bool:
        return self.get_last_user_intent() == 'greet'

    def process(self):
        caller = self.get_entity('caller')
        if caller is None:
            self.utter_one_of(
                "你好, 你需要什么",
                "你好, 有什么需要呢",
                "你好, 有什么可以帮你的呢",
            )
        else:
            import time
            time.sleep(3)
            self.utter_message("你好, 我是小智")
            self.set_slot('caller', caller)

