import datetime

from flask import Flask, request, json, jsonify

from src.annotation_handler import createNewAnnotation
from src.flask_setup import app
from src.predict import classifyDataset


@app.route('/hitec/classify/concepts/bi-lstm-classifier/run', methods=['POST'])
def classify_tore():

    app.logger.info('Bi-LSTM Classification run requested')
    app.logger.debug('/hitec/classify/concepts/bi-lstm-classifier/run called')

    content = json.loads(request.data.decode('utf-8'))

    documents = content["dataset"]["documents"]
    dataset_name = content["dataset"]["name"]
    annotation_name = content["params"]["annotation_name"]
    create = content["params"]["persist"] == 'true'

    app.logger.info(f'Create settings: {create}, {type(create)}')

    codes = classifyDataset(documents, app.logger)
    if create:
        createNewAnnotation(dataset_name, annotation_name, codes, app.logger)

    result = dict()
    result.update({"codes": codes})
    return jsonify(result)

@app.route('/hitec/classify/concepts/bi-lstm-classifier/status', methods=["GET"])
def get_status():
    status = {
        "status": "operational",
    }

    return jsonify(status)

