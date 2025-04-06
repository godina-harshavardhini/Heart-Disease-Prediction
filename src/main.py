from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Define input data schema
class InputData(BaseModel):
    features: list[list[float]]  # Expecting a 2D list

# Load the trained model
model_path = "/Applications/MLOPSproject/models/heart_disease_model.pkl"

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        print(f"✅ Model loaded successfully! Model expects {model.n_features_in_} features.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Ensure model is None if loading fails

@app.get("/")
def read_root():
    return {"message": "Welcome to the MLOps API!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        if model is None:
            return {"error": "Model is not loaded correctly."}
        
        input_features = np.array(data.features)

        # Check if input matches model feature count
        if input_features.shape[1] != model.n_features_in_:
            return {"error": f"Model expects {model.n_features_in_} features, but got {input_features.shape[1]}"}

        prediction = model.predict(input_features).tolist()
        return {"prediction": prediction}
    
    except Exception as e:
        return {"error": str(e)}
import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
