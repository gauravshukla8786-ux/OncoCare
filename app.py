from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("models/cancer_classifier.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    scaled = scaler.transform([data])
    prediction = model.predict(scaled)

    result = "Malignant" if prediction[0] == 1 else "Benign"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
