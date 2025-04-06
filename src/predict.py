import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_heart_disease(input_data):
    # Convert input data to DataFrame
    columns = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 
               'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 
               'diaBP', 'BMI', 'heartRate', 'glucose']
    
    df = pd.DataFrame([input_data], columns=columns)
    
    # Scale input data
    df_scaled = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]  # Probability of heart disease
    
    return {"Prediction": "Has Heart Disease" if prediction == 1 else "No Heart Disease",
            "Probability": f"{probability:.2%}"}

# Example usage
if __name__ == "__main__":
    sample_input = [1, 50, 2, 1, 20, 0, 0, 1, 0, 200, 130, 85, 25.5, 75, 110]
    result = predict_heart_disease(sample_input)
    print(result)
