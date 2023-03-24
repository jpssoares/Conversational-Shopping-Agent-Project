from flask import Flask, request
from flask_cors import CORS
import json

app = Flask(__name__) # create the Flask app
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)

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
        responseDict = { "has_response": True, "recommendations":"",
        "response":"Hello world!", "system_action":""}
        jsonString = json.dumps(responseDict)
    return jsonString

app.run(port=4000) # run app in debug mode on port 5000