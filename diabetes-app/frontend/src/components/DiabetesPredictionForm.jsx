import React from 'react'
import { useState } from 'react'
import axios from 'axios';

const DiabetesPredictionForm = () => {
    const [formData, setFormData] = useState({
        pregnancies: '',
        glucose: '',
        bloodPressure: '',
        skinThickness: '',
        insulin: '',
        bmi: '',
        diabetesPedigreeFunction: '',
        age: ''
    });

    const [prediction, setPrediction] = useState(null);

    const handleChange = (e) => {
        setFormData({...formData, [e.target.name]: e.target.value});
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
          const response = await axios.post('http://localhost:5000/predict', formData);
          setPrediction(response.data.prediction);
        } catch (error) {
          console.error("Prediction error:", error);
          alert("Prediction failed. Check the console for details.");
        }
      };

    return (
        <div className="min-h-screen bg-blue-50 flex flex-col items-center p-6">
        <h1 className="text-4xl font-bold text-blue-900 mb-2">Diabetes Risk Assessment</h1>
        <p className="text-sm text-blue-700 mb-6">Enter your health metrics below for a preliminary diabetes risk assessment</p>
  
        <form onSubmit={handleSubmit} className="bg-white p-8 rounded-lg shadow-md max-w-3xl w-full grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
          {[
            { name: "pregnancies", label: "Number of Pregnancies" },
            { name: "glucose", label: "Glucose Level (mg/dL)" },
            { name: "bloodPressure", label: "Blood Pressure (mm/Hg)" },
            { name: "skinThickness", label: "Skin Thickness (mm)" },
            { name: "insulin", label: "Insulin (mu U/ml)" },
            { name: "bmi", label: "BMI" },
            { name: "diabetesPedigree", label: "Diabetes Pedigree Function" },
            { name: "age", label: "Age" },
          ].map(({ name, label }) => (
            <div key={name}>
              <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
              <input
                name={name}
                type="text"
                value={formData[name]}
                onChange={handleChange}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder={`Enter ${label.toLowerCase()}`}
              />
            </div>
          ))}
  
          <div className="md:col-span-2">
            <button
              type="submit"
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-md transition"
            >
              Predict Risk
            </button>
          </div>
        </form>
  
        {prediction && (
          <div className="mb-6 text-lg font-medium text-green-700 bg-green-100 p-4 rounded-md shadow-sm">
            Prediction: {prediction}
          </div>
        )}
  
        <div className="bg-white p-6 rounded-lg shadow-md max-w-2xl text-left text-gray-700">
          <h2 className="text-lg font-semibold text-blue-800 mb-2">Important Health Information</h2>
          <p className="text-sm mb-3">
            This tool provides an initial assessment of diabetes risk based on key health indicators. Please note that this is not a medical diagnosis.
          </p>
          <ul className="list-disc pl-5 text-sm mb-3">
            <li>Fasting Glucose: 70-99 mg/dL</li>
            <li>Blood Pressure: Below 120/80 mm/Hg</li>
            <li>BMI: 18.5–24.9</li>
          </ul>
          <p className="text-xs italic text-gray-500">
            Always consult with healthcare professionals for proper medical advice and diagnosis.
          </p>
        </div>
      </div>
    )
}

export default DiabetesPredictionForm