# Diabetes Prediction System

## Overview

This project is a Machine Learning based Diabetes Prediction System developed using Python, Scikit-learn, and Gradio.

The application predicts whether a patient is diabetic or non-diabetic based on medical information entered by the user.

---

## Features

- Predicts diabetes using Machine Learning
- User-friendly Gradio interface
- K-Nearest Neighbors (KNN) algorithm
- Data preprocessing using StandardScaler
- Easy deployment using Render

---

## Dataset

The project uses the Pima Indians Diabetes Dataset.

Input Features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Output:

- 0 → Non-Diabetic
- 1 → Diabetic

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Gradio
- Joblib

---

## Project Structure

```
Diabetes-Prediction-System/
│
├── app.py
├── train_model.py
├── diabetes.csv
├── diabetes_model.pkl
├── scaler.pkl
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SameerBhardwaj79/Diabetes-Prediction-System.git
```

Go to the project folder:

```bash
cd Diabetes-Prediction-System
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the application:

```bash
python app.py
```

Open:

```
http://127.0.0.1:7860
```

---

## Machine Learning Workflow

1. Load Dataset
2. Split Dataset
3. Standardize Features
4. Train KNN Model
5. Save Model
6. Load Model
7. Predict Diabetes

---

## Model

Algorithm Used:

- K-Nearest Neighbors (KNN)

Accuracy:

Approximately **69.48%**

---

## Author

Sameer Bhardwaj