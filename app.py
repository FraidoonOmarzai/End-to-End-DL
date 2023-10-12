from flask import Flask, request, jsonify, render_template
import os
from PnemoniaDisease.utils.utils import decodeImage
from PnemoniaDisease.pipeline.prediction import PredictionPipeline


app = Flask(__name__, template_folder='webapp/templates')


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET','POST'])
def trainRoute():
    os.system("dvc repro")
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
def predictRoute():
    clApp = ClientApp()
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)