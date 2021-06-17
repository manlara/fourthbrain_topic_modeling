from flask import Flask, jsonify, request as flask_request, render_template
import json
import requests
import os

TOKEN = os.environ.get("NER_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
headers = {"Authorization": "Bearer {}".format(TOKEN)}

def query(payload):
	data = json.dumps(payload)
	response = requests.request("POST", API_URL, headers=headers, data=data)
	return json.loads(response.content.decode("utf-8"))

app = Flask(__name__)

@app.route("/inference/ner", methods=["POST"])
def ner_inference():
    payload = flask_request.get_json(force=True)
    print(payload)
    return jsonify(query(payload))

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")
# def home():
#     if flask_request.method == "GET":
#         return render_template("index.html")
#     if flask_request.method == "POST":
#         data = query({"inputs": "My name is Sarah Jessica Parker but you can call me Jessica"})
#     return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
