# OncoCare – Breast Cancer Detection System

OncoCare is a machine learning–based breast cancer detection system designed to classify tumors as **benign** or **malignant** using structured clinical data. The project focuses on building a clean, end-to-end ML pipeline covering data preprocessing, model training, evaluation, and prediction through a lightweight API.

---

## 🎯 Problem Statement

Early detection of breast cancer plays a crucial role in improving patient outcomes. However, manual interpretation of diagnostic data can be time-consuming and prone to inconsistency.  
OncoCare aims to support early diagnosis by applying machine learning techniques to analyze medical features and assist in tumor classification.

---

## 💡 Solution Overview

The system processes clinical data, applies preprocessing and feature scaling, and trains supervised machine learning models to predict whether a tumor is benign or malignant.  
A trained model is exposed via a simple Flask API, allowing predictions to be made programmatically without the need for a complex frontend.

---

## 🧠 Key Features

- Data preprocessing and feature scaling using Python  
- Supervised machine learning models for classification  
- Model evaluation using accuracy and classification metrics  
- Serialized trained model for reuse  
- REST API for real-time predictions  

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Machine Learning:** Scikit-learn  
- **Data Handling:** Pandas, NumPy  
- **Backend API:** Flask  
- **Model Persistence:** Pickle / Joblib  

---

## 🏗️ Project Structure

oncocare/
│
├── data/ # Dataset
├── src/ # ML pipeline modules
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ └── predict.py
│
├── models/ # Saved trained model
├── app.py # Flask API
├── requirements.txt
├── README.md


---

## 🔍 Machine Learning Workflow

1. Load and preprocess the dataset  
2. Perform feature scaling and train-test split  
3. Train classification models (Logistic Regression, Random Forest)  
4. Evaluate performance using accuracy and confusion matrix  
5. Save the best-performing model  
6. Serve predictions through a REST API  

---

## 🚀 API Usage

### POST `/predict`

**Request Body**
```json
{
  "features": [12.3, 15.6, 78.4, 450.2, ...]
}

```

## 📌 Why This Project Matters

OncoCare demonstrates practical application of machine learning fundamentals in a healthcare-related use case.
The project emphasizes clarity, reproducibility, and explainability over over-engineered interfaces, making it suitable for real-world ML workflows.

## 🔮 Future Improvements

Support for additional classification models
Improved evaluation metrics and visualization
Deployment to cloud platforms
Integration with a web-based UI

---


