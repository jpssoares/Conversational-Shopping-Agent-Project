from flask import Flask, request
from flask_cors import CORS
import json
import validators
import source.controller as ctrl
import source.conversation.dialog as dialog
import source.conversation.gpt as gpt
import source.conversation.product_qa as product_qa
import source.conversation.image_captioning as img_cap
from source.conversation.predefined_messages import *
from typing import ByteString

fst_message = True
last_results = None

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"
cors = CORS(app)


def interprete_msg(data) -> str:
    global fst_message
    global last_results
    input_msg: str = data.get("utterance")  # empty string if not present
    input_img = data.get("file")  # None if not present
    intent, keys, values = dialog.interpreter(input_msg)
    clothes = clothes_from_image(input_msg, input_img)
    if intent == "user_request_get_products" or (
        input_msg == "" and input_img is not None
    ):
        if clothes:
            match = img_cap.get_matching_clothes_quey(clothes, keys, values)
            print(match)
            if match is not None:
                input_msg = match
                search_type = "vqa_search"
        else:
            search_type = "text_search"

        last_results = ctrl.create_response_for_query(
            input_msg, input_img, keys, values, search_type
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

    elif intent in dialog.chat_intent_keys:
        gpt_answer = gpt.get_gpt_answer(input_msg)
        response = {
            "has_response": True,
            "recommendations": "",
            "response": gpt_answer,
            "system_action": "",
        }

    elif intent in dialog.qa_intent_keys:
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


def clothes_from_image(input_msg: str, input_img: ByteString):
    clothes = list()
    for part in input_msg.split():
        if validators.url(part):
            caption = img_cap.get_caption_for_image(part)
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


def _update_search_type(input_msg) -> dict:
    """
    Legacy code.
    """
    input_msg_parts = input_msg.split(" ")
    response = None
    if input_msg_parts[0] == "change_search_type":
        if input_msg_parts[1] in ctrl.search_types:
            ctrl.search_type = input_msg_parts[1]
            response = {
                "has_response": True,
                "recommendations": "",
                "response": MSG_SEARCH_TYPE_CHANGED,
                "system_action": "",
            }
        else:
            response = {
                "has_response": True,
                "recommendations": "",
                "response": MSG_SEARCH_TYPE_CHANGE_FAILED,
                "system_action": "",
            }

    return response


@app.route("/", methods=["POST"])
def dialog_turn():
    if request.is_json:
        data = request.json
        response = interprete_msg(data)
    return response


app.run(port=4000)
