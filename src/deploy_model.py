import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model and scaler
model = joblib.load("../models/heart_disease_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON input
    features = np.array(data["features"]).reshape(1, -1)  # Convert input to array
    features_scaled = scaler.transform(features)  # Scale input

    prediction = model.predict(features_scaled)[0]  # Predict
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
