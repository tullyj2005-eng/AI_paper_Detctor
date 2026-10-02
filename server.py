from flask import Flask, request, jsonify
from detector.predict import Detector

app = Flask(__name__)
detector = Detector()        # loads models/detector.joblib once

@app.post("/predict")
def predict():
    text = request.get_json()["text"]
    result = detector.predict(text)          # the same predict the Gradio app uses
    return jsonify(prob_ai=result["prob_ai"])

if __name__ == "__main__":
    app.run(port=5000)