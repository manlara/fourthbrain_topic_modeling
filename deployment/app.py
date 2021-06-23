from flask import Flask, jsonify, request, render_template
import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.environ.get("NER_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
headers = {"Authorization": "Bearer {}".format(TOKEN)}

app = Flask(__name__)


def query(payload):
    data = json.dumps(payload)
    response = requests.post(API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


@app.route("/legend/ner")
def ner_legend():
    legend = {
        "O": {
            "descr": "Outside of a named entity",
            "color": "light",
        },
        "MISC": {
            "descr": "Beginning of a miscellaneous entity right after another miscellaneous entity",
            "color": "warning",
        },
        "PER": {
            "descr": "Beginning of a person’s name right after another person’s name",
            "color": "success",
        },
        "ORG": {
            "descr": "Beginning of an organization right after another organization",
            "color": "primary",
        },
        "LOC": {
            "descr": "Beginning of a location right after another location",
            "color": "warning",
        },
    }
    return jsonify(legend)

@app.route("/inference/ner", methods=["POST"])
def ner_inference():
    payload = request.get_json(force=True)
    print(payload)
    return jsonify(query(payload))


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = query(
            {"inputs": "My name is Sarah Jessica Parker but you can call me Jessica"}
        )
        return jsonify(data)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
