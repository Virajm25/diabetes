from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app, origins='*')

# Load your trained ML model
model = model = joblib.load('diabetes_model_rf.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[ 
        float(data['pregnancies']),
        float(data['glucose']),
        float(data['bloodPressure']),
        float(data['skinThickness']),
        float(data['insulin']),
        float(data['bmi']),
        float(data['diabetesPedigree']),  # Match this to frontend
        float(data['age'])
    ]])

    prediction = model.predict(features)[0]
    output = 'High Risk' if prediction == 1 else 'Low Risk'

    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True)
