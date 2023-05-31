# from langdetect import detect
# from translate import Translator
from flask import Flask, request
from flask_cors import CORS
import json
import validators
import source.controller as ctrl
import source.conversation.dialog as dialog
import source.conversation.gpt as gpt
import source.conversation.product_qa as product_qa
import source.conversation.image_captioning as img_cap

# Program variables
beginning_msg = "Hello! Welcome to Farfetch! What item are you looking for?"
goodbye_msg = "Goodbye! If you need anything, I'll be here..."
retry_msg = (
    "Sorry, I did not understand what you were trying to tell me... can we try again?"
)
error_msg = "Sorry can't help you with that. Please try again..."
success_search_msg = "Here are some items I found..."
bad_search_msg = "Sorry, I couldn't find any products that meet your query..."
help_msg = (
    "Here are some commands you can use:\n"
    + "Change the search type: change_search_type <search_type> (full_text, boolean_search, text_and_attrs, emb_search)\n"
    + "Search for product using boolean filtering: must <field1> a ... <field2> b should <field3> c must_not <field4> d filter <field5> e\n"
    + "Search for Products with Text and Attributes\n<field> <query>\nExample: product_main_colour black\n"
    + "Searching for Products with Cross-Modal Spaces\n<query_w1> <query_w2>\nExample: black boots\n"
)

search_type_changed_msg = "The search type was successfully changed"
search_type_change_error = "That search type doesn't exist...\nTry another one"

fst_message = True
last_results = None
current_lang = "en"

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"
cors = CORS(app)

def translate_to_lang(msg, lang=current_lang):
    global current_lang
    if current_lang == 'en':
        return msg
    
    # FIXME: implement translate logic here
    return msg


def interprete_msg(data):
    global current_lang
    global fst_message
    global last_results
    input_msg = data.get("utterance")
    input_img = data.get("file")

    jsonString = ""

    # check if user is still responding in the same language
    # FIXME: implement language detection here
    # if not fst_message:
    #     current_lang = detect(input_msg)

    # in order to maintain the functionality from the first part,
    # we leave the change_search_type as is
    input_msg_parts = input_msg.split(" ")
    if input_msg_parts[0] == "change_search_type":
        if input_msg_parts[1] in ctrl.search_types:
            ctrl.search_used = input_msg_parts[1]
            responseDict = {
                "has_response": True,
                "recommendations": "",
                "response": translate_to_lang(search_type_changed_msg),
                "system_action": "",
            }
        else:
            responseDict = {
                "has_response": True,
                "recommendations": "",
                "response": translate_to_lang(search_type_change_error),
                "system_action": "",
            }
    

    translated_input = translate_to_lang(input_msg, "en")
    intent, keys, values = dialog.interpreter(translated_input)
    print(intent)
    # print(keys)
    # print(values)
    
    clothes = []
    if validators.url(input_msg_parts[0]):
        # get image caption
        caption = img_cap.get_caption_for_image(input_msg_parts[0])
        print("caption: " + caption)
        # get clothes array using GPT
        if caption != None:
            clothes = img_cap.get_clothing_items_from_caption(caption)

    if intent == "user_request_get_products" or (input_msg == "" and input_img != None):
        if clothes != []:
            match = img_cap.get_matching_clothes_quey(clothes, keys, values)
            if match != None:
                input_msg = match
                ctrl.search_used = "vqa_search"

        last_results = ctrl.create_response_for_query(
            input_msg, input_img, keys, values
        )
        if last_results == None:
            responseDict = {
                "has_response": True,
                "recommendations": None,
                "response": translate_to_lang(bad_search_msg),
                "system_action": "",
            }
        else:
            responseDict = {
                "has_response": True,
                "recommendations": last_results,
                "response": translate_to_lang(success_search_msg),
                "system_action": "",
            }

    elif intent == "user_neutral_greeting":
        fst_message = False
        responseDict = {
            "has_response": True,
            "recommendations": "",
            "response": translate_to_lang(beginning_msg),
            "system_action": "",
        }

    # dont translate help message
    elif intent == "user_neutral_what_can_i_ask_you":
        responseDict = {
            "has_response": True,
            "recommendations": "",
            "response": help_msg,
            "system_action": "",
        }

    elif intent == "user_neutral_goodbye":
        responseDict = {
            "has_response": True,
            "recommendations": "",
            "response": translate_to_lang(goodbye_msg),
            "system_action": "",
        }

    # dont translate text generated by GPT(it does it automatically)
    elif intent in dialog.chat_intent_keys:
        gpt_answer = gpt.get_gpt_answer(input_msg)
        responseDict = {
            "has_response": True,
            "recommendations": "",
            "response": gpt_answer,
            "system_action": "",
        }

    elif intent in dialog.qa_intent_keys:
        answer = product_qa.get_qa_answer(intent, last_results, input_msg)
        responseDict = {
            "has_response": True,
            "recommendations": "",
            "response": answer,
            "system_action": "",
        }

    else:
        fst_message = False
        responseDict = {
            "has_response": True,
            "recommendations": "",
            "response": translate_to_lang(error_msg),
            "system_action": "",
        }

    jsonString = json.dumps(responseDict)
    return jsonString


@app.route("/", methods=["POST"])
def dialog_turn():
    if request.is_json:
        data = request.json
        # print(data)
        jsonString = interprete_msg(data)
    return jsonString


app.run(port=4000)
