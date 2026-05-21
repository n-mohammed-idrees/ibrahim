from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import traceback

app = Flask(__name__)
CORS(app)
@app.route("/")
def home():
    return "Backend Running Successfully"
@app.route("/predict", methods=["POST"])
def predict():
    try:
        model = joblib.load("diabetes_model.pkl")
        scaler = joblib.load("diabetes_scaler.pkl")
        data = request.get_json(force=True)
        x = np.array(data, dtype=float).reshape(1, -1)
        x = scaler.transform(x)
        prediction = model.predict(x)
        return jsonify({
            "prediction": int(prediction[0])
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
@app.route("/carpredict", methods=["POST"])
def carpredict():
    try:
        model = joblib.load("Carmodel.pkl")
        data = request.get_json(force=True)
        x = np.array(data, dtype=float).reshape(1, -1)
        prediction = model.predict(x)
        return jsonify({
            "prediction": float(prediction[0])
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500  
app.run()      