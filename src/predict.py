import joblib
import numpy as np

def predict(sample_input):
    model = joblib.load("models/cancer_classifier.pkl")
    scaler = joblib.load("models/scaler.pkl")

    sample_scaled = scaler.transform([sample_input])
    prediction = model.predict(sample_scaled)

    return "Malignant" if prediction[0] == 1 else "Benign"

if __name__ == "__main__":
    sample = [14.2, 20.1, 92.3, 654.9]  # example features
    print("Prediction:", predict(sample))
