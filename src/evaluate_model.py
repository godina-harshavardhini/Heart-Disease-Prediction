import joblib
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_preprocess_data

def evaluate_model():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    
    # Load model
    model = joblib.load("../models/heart_disease_model.pkl")

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    evaluate_model()
