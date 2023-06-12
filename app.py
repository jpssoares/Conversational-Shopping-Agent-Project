from flask import Flask, request
from flask_cors import CORS
import json
import validators
import source.controller as ctrl
import source.conversation.dialog as dialog
import source.conversation.product_qa as product_qa
import source.conversation.image_captioning as img_cap
from source.conversation.ordinals import get_position
from source.conversation.predefined_messages import *
from typing import ByteString


fst_message = True
last_results = None
provided_characteristics = dict()
NECESSARY_CHARACTERISTICS = ["category", "colour"]
missing_characteristics = list()

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"
cors = CORS(app)


def interpret_msg(data: dict) -> str:
    global fst_message
    global last_results
    global provided_characteristics
    global missing_characteristics
    input_msg: str = data.get("utterance")  # empty string if not present
    input_img = data.get("file")  # None if not present
    intent, slots, values = dialog.interpreter(input_msg)
    slots, values = _update_provided_characteristics(slots, values)
    ordinal = get_position(input_msg)

    print(
        f"Message '{input_msg}', intent '{intent}', provided characteristics '{provided_characteristics}'"
    )

    if (
        _asking_about_more_similar_products(ordinal, last_results)
        and intent not in dialog.QA_INTENT_KEYS
    ):
        print(f"Returning more products like {last_results[ordinal]}")
        last_results = ctrl.get_similar(last_results[ordinal])
        response = {
            "has_response": True,
            "recommendations": last_results,
            "response": SUCCESS_SEARCH_MSG,
            "system_action": "",
        }
        return json.dumps(response)

    if intent == "user_request_get_products":
        links_to_images = [part for part in input_msg.split() if validators.url(part)]
        if (input_msg == "" and input_img is not None) or links_to_images:
            print("Looking for clothes for VQA")
            clothes = clothes_from_image(links_to_images, input_img)
            if clothes:
                print("Found clothes on the image")
                match = img_cap.get_matching_clothes_quey(clothes, slots, values)
                if match is not None:
                    print("Found a match, creating response based on VQA")
                    last_results = ctrl.create_response_for_query(
                        match, input_img, slots, values, search_type="vqa_search"
                    )
                    if last_results is None:
                        response = {
                            "has_response": True,
                            "recommendations": last_results,
                            "response": SUCCESS_SEARCH_MSG,
                            "system_action": "",
                        }
                    else:
                        response = {
                            "has_response": True,
                            "recommendations": None,
                            "response": BAD_SEARCH_MSG,
                            "system_action": "",
                        }
                    return json.dumps(response)

        _update_missing_characteristics()
        if missing_characteristics:
            print(f"Asking for additional information about: {missing_characteristics}")
            response = {
                "has_response": True,
                "recommendations": None,
                "response": missing_characteristics_response(missing_characteristics),
                "system_action": "",
            }
            return json.dumps(response)
        else:
            provided_characteristics = dict()
            missing_characteristics = list()

        print("Creating a response based just on embeddings")
        last_results = ctrl.create_response_for_query(
            input_msg, input_img, slots, values
        )
        if last_results is None:
            response = {
                "has_response": True,
                "recommendations": None,
                "response": BAD_SEARCH_MSG,
                "system_action": "",
            }
        else:
            response = {
                "has_response": True,
                "recommendations": last_results,
                "response": SUCCESS_SEARCH_MSG,
                "system_action": "",
            }

    elif intent == "user_neutral_greeting":
        fst_message = False
        response = {
            "has_response": True,
            "recommendations": "",
            "response": BEGGINING_MSG,
            "system_action": "",
        }
    elif intent == "user_neutral_what_can_i_ask_you":
        response = {
            "has_response": True,
            "recommendations": "",
            "response": HELP_MSG,
            "system_action": "",
        }
    elif intent == "user_neutral_goodbye":
        response = {
            "has_response": True,
            "recommendations": "",
            "response": GOODBYE_MSG,
            "system_action": "",
        }
    elif intent in dialog.QA_INTENT_KEYS:
        answer = product_qa.get_qa_answer(intent, last_results, input_msg)
        response = {
            "has_response": True,
            "recommendations": "",
            "response": answer,
            "system_action": "",
        }
    else:
        fst_message = False
        response = {
            "has_response": True,
            "recommendations": "",
            "response": ERROR_MSG,
            "system_action": "",
        }

    return json.dumps(response)


def _asking_about_more_similar_products(ordinal: int, last_results) -> bool:
    return last_results is not None and ordinal is not None


def _update_provided_characteristics(
    slots: list[str], values: list[str]
) -> tuple[list[str], list[str]]:
    global missing_characteristics
    global provided_characteristics

    if missing_characteristics and not slots:
        # If there were some characteristics missing, but no comprehensible response was provided answer is assumed to be "any"
        for characteristic in missing_characteristics:
            provided_characteristics[characteristic] = ""

    # we use all previously provided characteristics, but if user changed their mind newest value is used
    for slot, value in zip(slots, values):
        provided_characteristics[slot] = value
    slots = list(provided_characteristics.keys())
    values = list(provided_characteristics.values())

    return slots, values


def _update_missing_characteristics():
    global missing_characteristics
    missing_characteristics = [
        characteristic
        for characteristic in NECESSARY_CHARACTERISTICS
        if characteristic not in provided_characteristics.keys()
    ]


def clothes_from_image(links_to_images: list[str], input_img: ByteString):
    clothes = list()
    for link in links_to_images:
        caption = img_cap.get_caption_for_image(link)
        print("caption: " + str(caption))
        if caption is not None:
            clothes_per_link = img_cap.get_clothing_items_from_caption(caption)
            clothes.extend(clothes_per_link)
    if input_img is not None:
        caption = img_cap.get_caption_for_image(input_img)
        print("caption: " + str(caption))
        if caption is not None:
            clothes_per_link = img_cap.get_clothing_items_from_caption(caption)
            clothes.extend(clothes_per_link)
    return clothes


@app.route("/", methods=["POST"])
def dialog_turn():
    if request.is_json:
        data = request.json
        response = interpret_msg(data)
    return response


app.run(port=4000)
