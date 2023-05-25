from flask import Flask, request
from flask_cors import CORS
import json
import source.controller as ctrl
import source.dialog as dialog
import openai
import os

# Program variables
openai.api_key = ""
beginning_msg = "Hello! Welcome to Farfetch! What item are you looking for?"
goodbye_msg = "Goodbye! If you need anything, I'll be here..."
retry_msg = (
    "Sorry, I did not understand what you were trying to tell me... can we try again?"
)
error_msg = "Sorry can't help you with that. Please try again..."
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

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"
cors = CORS(app)

def run_with_gpt_setup():
    selected = input("Would you like to run using GPT-3? yes/[no] ")
    if selected.upper() == "YES":
        openai.api_key = os.getenv("OPENAI_API_KEY")

def get_gpt_answer(msg="What is Farfetch?"):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", # this is "ChatGPT" $0.002 per 1k tokens
        messages=[{"role": "user", "content": msg}]
    )

    reply_content = completion.choices[0].message.content
    return reply_content


def interprete_msg(data):
    global fst_message
    global last_results
    input_msg = data.get("utterance")
    input_img = data.get("file")


    jsonString = ""

    # in order to maintain the functionality from the first part,
    # we leave the change_search_type as is
    input_msg_parts = input_msg.split(" ")
    if input_msg_parts[0] == "change_search_type":
        if input_msg_parts[1] in ctrl.search_types:
            ctrl.search_used = input_msg_parts[1]
            responseDict = {
                "has_response": True,
                "recommendations": "",
                "response": search_type_changed_msg,
                "system_action": "",
            }
        else:
            responseDict = {
                "has_response": True,
                "recommendations": "",
                "response": search_type_change_error,
                "system_action": "",
            }

    intent, keys, values = dialog.interpreter(input_msg)
    print(intent)

    if intent == "user_request_get_products" or (input_msg=="" and input_img!=None):
        responseDict, last_results = ctrl.create_response_for_query(input_msg, input_img, keys, values)
        print(last_results[0])

    elif intent == "user_neutral_greeting":
        fst_message = False
        responseDict = {
            "has_response": True,
            "recommendations": "",
            "response": beginning_msg,
            "system_action": "",
        }

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
            "response": goodbye_msg,
            "system_action": "",
        }
    
    elif intent in dialog.chat_intent_keys:
       gpt_answer = get_gpt_answer(input_msg)
       responseDict = {
            "has_response": True,
            "recommendations": "",
            "response": gpt_answer,
            "system_action": "",
        }
    
    elif intent in dialog.qa_intent_keys:
       responseDict = {
            "has_response": True,
            "recommendations": "",
            "response": dialog.get_qa_answer(),
            "system_action": "",
        }

    else:
        fst_message = False
        responseDict = {
            "has_response": True,
            "recommendations": "",
            "response": error_msg,
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

run_with_gpt_setup()
app.run(port=4000)
