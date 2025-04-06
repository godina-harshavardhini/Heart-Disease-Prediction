import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("/Applications/MLOPSproject/data/heart_disease.csv")  # Ensure this is your dataset file

# Separate features (X) and target variable (y)
X = df.drop(columns=["CVD"])  # ❌ Remove "CVD" before training
y = df["CVD"]  # Define target variable
print(f"Training features shape: {X.shape}")
print(X.columns)


# Train model
model = RandomForestClassifier()
model.fit(X, y)  # ✅ Model is now trained on correct features

# Save model
model_path = "/Applications/MLOPSproject/models/heart_disease_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
print(f"Training features shape: {X.shape}")  # Should now be (4240, 14)
print(f"Feature names used for training: {list(X.columns)}")  # Should be 14 features
