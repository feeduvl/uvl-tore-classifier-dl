import datetime

from flask import Flask, request, json, jsonify

from src.flask_setup import app


@app.route('/hitec/classify/concepts/bi-lstm-classifier/run', methods=['POST'])
def classify_tore():

    app.logger.info('Bi-LSTM Classification run requested')
    app.logger.debug('/hitec/classify/concepts/bi-lstm-classifier/run called')
    timestamp = '{:%Y-%m-%d_%H%M%S-%f}'.format(datetime.datetime.now())

    content = json.loads(request.data.decode('utf-8'))
    app.logger.info(content)

    documents = content["dataset"]["documents"]
    dataset_name = content["dataset"]["name"]
    annotation_name = content["params"]["annotation_name"]
    create = content["params"]["persist"] == 'true'

    app.logger.info(f'Create settings: {create}, {type(create)}')

    # annotation_handler = AnnotationHandler(annotation_name, dataset_name, app.logger)
    # request_handler = RequestHandler(app.logger, annotation_handler)
    # codes = request_handler.process(documents, create)
    #
    # result = dict()
    # result.update({"codes": codes})
    # return jsonify(result)

@app.route('/hitec/classify/concepts/bi-lstm-classifier/status', methods=["GET"])
def get_status():
    status = {
        "status": "operational",
    }

    return jsonify(status)

