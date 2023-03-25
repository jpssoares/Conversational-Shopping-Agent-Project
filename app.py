from flask import Flask, request
from flask_cors import CORS
import json
import pprint as pp
import requests
import json
from tqdm import tqdm
import pprint as pp
from opensearchpy import OpenSearch
from opensearchpy import helpers
from PIL import Image
import pandas as pd
import time
import numpy as np
import controller as ctrl

#from transformers import CLIPProcessor, CLIPModel
import transformers as tt

# Program variables
beginning_msg = "Hello! Welcome to Farfetch! What item are you looking for?"
retry_msg = "Sorry, I did not understand what you were trying to tell me... can we try again?"

help_msg = "Here are some commands you can use:\n" \
    + "Search for Products with Text and Attributes\n<field> <query>\nExample: product_main_colour black\n" \
    + "Searching for Products with Cross-Modal Spaces\n<query_w1> <query_w2>\nExample: black boots\n"

# Program initiation
app = Flask(__name__) # create the Flask app
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)


def interprete_msg(data):
    input_msg = data.get('utterance')
    jsonString = ''
    
    if (input_msg=='Hi!'):
        responseDict = { "has_response": True, "recommendations":"", "response":beginning_msg, "system_action":""}
    elif(input_msg.lower()=='help'):
        responseDict = { "has_response": True, "recommendations":"", "response":help_msg, "system_action":""}
    else:
        responseDict = ctrl.create_response_for_query(input_msg)
    jsonString = json.dumps(responseDict)
    return jsonString


@app.route('/', methods=['POST'])
def dialog_turn():
    if request.is_json:
        data = request.json
        print(data)
        print(data.get('utterance'))
        print(data.get('session_id'))
        print(data.get('user_action'))
        print(data.get('interface_selected_product_id'))
        print(data.get('image'))
        jsonString = interprete_msg(data)
    return jsonString

app.run(port=4000) # run app in debug mode on port 5000