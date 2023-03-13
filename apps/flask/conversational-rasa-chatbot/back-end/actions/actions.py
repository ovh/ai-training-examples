# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
#
from rasa_sdk import Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []


class ValidateBilan(FormValidationAction):
    def name(self) -> Text:
        return "validate_bilan"

    @staticmethod
    def type_of_house() -> List[Text]:
        return ["appartment", "house", "flat"]

    def validate_typeofhouse(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        if slot_value.lower() in self.type_of_house():
            return {"typeofhouse": slot_value.lower()}
        else:
            return {"typeofhouse": None}

    @staticmethod
    def way_to_move() -> List[Text]:
        return ["metro", "rer", "car", "motorbike", "bike", "on foot", "common transport","bus"]

    def validate_typeoftransport(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        if slot_value.lower() in self.way_to_move():
            return {"typeoftransport": slot_value.lower()}
        else:
            return {"typeoftransport": None}

    @staticmethod
    def way_of_working() -> List[Text]:
        return ["remote working", "on site", "teleworking", "on a building"]

    def validate_typeofwork(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        if slot_value.lower() in self.way_of_working():
            return {"typeofwork": slot_value.lower()}
        else:
            return {"typeofwork": None}
