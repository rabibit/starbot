# coding: utf8
from rasa_sdk import Action
from rasa_sdk.events import AllSlotsReset
from rasa_sdk.forms import FormAction
import typing

if typing.TYPE_CHECKING:
    from typing import Text
    from rasa_core_sdk.executor import CollectingDispatcher, Tracker


class ProcessIntentAction(Action):
    def name(self):
        return "action_process_intent"

    def run(self, dispatcher, tracker, domain):
        pass

