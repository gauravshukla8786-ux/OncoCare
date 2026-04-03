from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_preprocessing import load_and_preprocess_data

def train_model():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        "data/breast_cancer.csv"
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    joblib.dump(model, "models/cancer_classifier.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Model trained and saved successfully.")

if __name__ == "__main__":
    train_model()
