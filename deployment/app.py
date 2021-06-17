from flask import Flask, jsonify, request, render_template
import json
import requests
import os

TOKEN = os.environ.get("NER_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
headers = {"Authorization": "Bearer {}".format(TOKEN)}

app = Flask(__name__)


def query(payload):
	data = json.dumps(payload)
	response = requests.post(API_URL, headers=headers, data=data)
	return json.loads(response.content.decode("utf-8"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = query({"inputs": "My name is Sarah Jessica Parker but you can call me Jessica"})
        return jsonify(data)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
