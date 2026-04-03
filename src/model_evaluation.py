from sklearn.metrics import classification_report, confusion_matrix
import joblib
from data_preprocessing import load_and_preprocess_data

def evaluate_model():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(
        "data/breast_cancer.csv"
    )

    model = joblib.load("models/cancer_classifier.pkl")
    predictions = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    evaluate_model()
