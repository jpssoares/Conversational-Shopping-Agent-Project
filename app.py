from flask import Flask, request
from flask_cors import CORS
import json
import controller as ctrl

# Program variables
beginning_msg = "Hello! Welcome to Farfetch! What item are you looking for?"
retry_msg = (
    "Sorry, I did not understand what you were trying to tell me... can we try again?"
)

help_msg = (
    "Here are some commands you can use:\n"
    + "Change the search type: change_search_type <search_type> (full_text, boolean_search, text_and_attrs, emb_search)\n"
    + "Search for product using boolean filtering: must <field1> a ... <field2> b should <field3> c must_not <field4> d filter <field5> e\n"
    + "Search for Products with Text and Attributes\n<field> <query>\nExample: product_main_colour black\n"
    + "Searching for Products with Cross-Modal Spaces\n<query_w1> <query_w2>\nExample: black boots\n"
)

search_type_changed_msg = "The search type was successfully changed"
search_type_change_error = "That search type doesn't exist...\nTry another one"

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"
cors = CORS(app)


def interprete_msg(data):
    input_msg = data.get("utterance")
    input_msg_parts = input_msg.split(" ")
    jsonString = ""

    if input_msg == "Hi!":
        responseDict = {
            "has_response": True,
            "recommendations": "",
            "response": beginning_msg,
            "system_action": "",
        }
    elif input_msg.lower() == "help":
        responseDict = {
            "has_response": True,
            "recommendations": "",
            "response": help_msg,
            "system_action": "",
        }
    elif input_msg_parts[0] == "change_search_type":
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
    else:
        responseDict = ctrl.create_response_for_query(input_msg)
    jsonString = json.dumps(responseDict)
    return jsonString


@app.route("/", methods=["POST"])
def dialog_turn():
    if request.is_json:
        data = request.json
        print(data)
        jsonString = interprete_msg(data)
    return jsonString


app.run(port=4000)
