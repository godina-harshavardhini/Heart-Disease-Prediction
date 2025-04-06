from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define paths
model_path = os.path.join("models", "heart_disease_model.pkl")
scaler_path = os.path.join("models", "scaler.pkl")

# Ensure "models" directory exists
os.makedirs("models", exist_ok=True)

# Check and create scaler if not exists
def check_and_create_scaler():
    if not os.path.exists(scaler_path):
        print("⚠️ Scaler file not found. Creating a new one...")
        
        df = pd.read_csv("data/Heart_disease.csv")
        X = df.drop(columns=["CVD"])  # Exclude target column

        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, scaler_path)
        print("✅ Scaler saved successfully!")

# Ensure model and scaler exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}")

check_and_create_scaler()

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print("✅ Model and Scaler loaded successfully.")

# Initialize FastAPI
app = FastAPI()

# Define input model to accept "features" key
class HeartDiseaseInput(BaseModel):
    features: list[list[float]]

@app.post("/predict")
def predict_heart_disease(data: HeartDiseaseInput):
    try:
        # Convert input to NumPy array
        input_data = np.array(data.features)

        # Scale input data
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running!"}
