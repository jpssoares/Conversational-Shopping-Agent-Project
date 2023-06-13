from PIL.Image import Image
from flask import Flask, request
from flask_cors import CORS
import json
import os
import source.controller as ctrl
import source.conversation.dialog as dialog
import source.conversation.product_qa as product_qa
import source.conversation.image_captioning as img_cap
from source.conversation.predefined_messages import *
from source.image_handling import load_image
from source.conversation.text_processing import get_position, preprocess_input_msg

fst_message = True
last_results = None
provided_characteristics = dict()
NECESSARY_CHARACTERISTICS = ["category"]
missing_characteristics = list()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"
cors = CORS(app)


def interprete_msg(data: dict) -> str:
    global fst_message
    global last_results
    global provided_characteristics
    global missing_characteristics
    input_msg: str = data.get("utterance")  # empty string if not present
    input_img = data.get("file")  # None if not present
    input_img = load_image(input_msg, input_img)

    intent, slots, values = dialog.interpreter(input_msg)
    slots, values, provided_characteristics = update_provided_characteristics(
        slots, values, provided_characteristics, missing_characteristics
    )
    ordinal = get_position(input_msg)
    input_msg = preprocess_input_msg(input_msg, values)
    print(
        f"Processed message: '{input_msg}', intent: '{intent}', provided characteristics: '{provided_characteristics}'",
        f"Detected slots: '{slots}', values: '{values}', ordinal: {ordinal}",
        sep="\n",
    )
    if (
        last_results is not None
        and ordinal is not None
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

    if (
        intent == "user_request_get_products"
        or (input_msg == "" and input_img is not None)
        or missing_characteristics
    ):

        clothes = clothes_from_image(input_img)
        print(f"Trying VQA, found clothes: {clothes}")
        if clothes:
            match = img_cap.get_matching_clothes_quey(clothes, slots, values)
            if match is not None:
                input_msg = match
                search_type = "vqa_search"
                print(f"Searching with VQA for '{input_msg}'")
            else:
                search_type = "text_search"
        else:
            print(f"Clothes on image not found")
            search_type = "text_search"

        missing_characteristics = update_missing_characteristics(
            provided_characteristics
        )
        if missing_characteristics:
            print(
                f"Asking for information about missing characteristics: {missing_characteristics}"
            )
            response = response = {
                "has_response": True,
                "recommendations": None,
                "response": missing_characteristics_response(missing_characteristics),
                "system_action": "",
            }
            return json.dumps(response)
        else:
            missing_characteristics = list()
            try:
                provided_characteristics = {
                    "category_gender_name": provided_characteristics.get(
                        "category_gender_name"
                    )
                }
            except KeyError:
                provided_characteristics = dict()

        last_results = ctrl.create_response_for_query(
            input_msg, input_img, slots, values, search_type
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
    elif intent == "user_neutral_are_you_a_bot":
        response = {
            "has_response": True,
            "recommendations": "",
            "response": ARE_YOU_A_BOT_MSG,
            "system_action": "",
        }
    elif intent == "user_neutral_what_is_your_name":
        response = {
            "has_response": True,
            "recommendations": "",
            "response": WHO_ARE_YOU_MSG,
            "system_action": "",
        }
    elif intent == "user_neutral_who_do_you_work_for":
        response = {
            "has_response": True,
            "recommendations": "",
            "response": WHO_DO_YOU_WORK_FOR_MSG,
            "system_action": "",
        }
    elif intent == "user_neutral_who_made_you":
        response = {
            "has_response": True,
            "recommendations": "",
            "response": WHO_MADE_YOU_MSG,
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


def update_provided_characteristics(
    slots: list[str],
    values: list[str],
    provided_characteristics: dict[str, str],
    missing_characteristics: list[str],
):
    if missing_characteristics and not slots:
        # If there were some characteristics missing, but no comprehensible response was provided answer is assumed to be "any".
        for characteristic in missing_characteristics:
            provided_characteristics[characteristic] = ""

    # we use all previously provided characteristics, but if user changed their mind newest value is used
    for slot, value in zip(slots, values):
        if slot == "dress_silhouette":
            slot = "category"
        provided_characteristics[slot] = value

    updated_slots = list(provided_characteristics.keys())
    update_values = list(provided_characteristics.values())

    return updated_slots, update_values, provided_characteristics


def update_missing_characteristics(provided_characteristics: dict[str, str]):
    return [
        characteristic
        for characteristic in NECESSARY_CHARACTERISTICS
        if characteristic not in provided_characteristics.keys()
    ]


def clothes_from_image(input_img: Image):
    clothes = list()
    if input_img is not None:
        print("input_img: " + str(input_img))
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
        response = interprete_msg(data)
    return response


app.run(port=4000)
