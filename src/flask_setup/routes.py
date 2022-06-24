import datetime

from flask import Flask, request, json, jsonify
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

from src.flask_setup import app


@app.route('/hitec/classify/concepts/deep-ner/run', methods=['POST'])
def classify_tore():

    app.logger.info('Deep-NER Classification run requested')
    app.logger.debug('/hitec/classify/concepts/deep-ner/run called')
    timestamp = '{:%Y-%m-%d_%H%M%S-%f}'.format(datetime.datetime.now())

    content = json.loads(request.data.decode('utf-8'))
    # app.logger.info(content)

    dataset = content["dataset"]["documents"]

    # Send request to get selected annotation
    # Ths should be a parameter when running algorithm

    # app.logger.info(dataset)

    return 'OK'


@app.route('/hitec/classify/concepts/deep-ner/status', methods=["GET"])
def get_status():
    status = {
        "status": "operational",
    }

    return jsonify(status)

