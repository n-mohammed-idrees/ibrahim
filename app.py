from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import traceback

app = Flask(__name__)

# Enable CORS for Flet web
CORS(app)


@app.route("/")
def home():
    return "Backend Running Successfully"


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Load model and scaler
        model = joblib.load("diabetes_model.pkl")
        scaler = joblib.load("diabetes_scaler.pkl")

        # Get JSON data
        data = request.get_json()

        print("Received Data:", data)

        # Convert to numpy array
        x = np.array(data, dtype=float).reshape(1, -1)

        print("Numpy Array:", x)

        # Scale data
        x = scaler.transform(x)

        print("Scaled Data:", x)

        # Predict
        prediction = model.predict(x)

        print("Prediction:", prediction)

        # Return response
        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:

        print(traceback.format_exc())

        return jsonify({
            "error": str(e)
        }), 500