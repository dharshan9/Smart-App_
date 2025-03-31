import os
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load trained LSTM model and scaler
model = tf.keras.models.load_model("health_risk_model.h5")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "\U0001F3E5 AI Health Tracker API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Receive JSON request
        
        # Extract input features
        heart_rate = float(data["heart_rate"])
        sleep_duration = float(data["sleep_duration"])
        sleep_category = int(data["sleep_category"])  # Unused in this version
        
        # Normalize inputs
        input_data = scaler.transform([[heart_rate, sleep_duration]])
        
        # Reshape for LSTM
        input_data = np.reshape(input_data, (1, 1, 2))  # (batch_size, time_steps, features)
        
        # Get prediction
        prediction = model.predict(input_data)[0][0]
        risk_status = "At Risk" if prediction > 0.5 else "Healthy"
        
        return jsonify({"prediction": float(prediction), "status": risk_status})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    print("✅ Flask App is Running on http://127.0.0.1:5000/")  # Debug print
    app.run(host="0.0.0.0", port=5000, debug=True)
