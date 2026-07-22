import gradio as gr
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age,
):
    data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    data = scaler.transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        return "🩺 Diabetic"
    else:
        return "✅ Non-Diabetic"


demo = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Diabetes Prediction System",
    description="Enter patient details to predict whether the patient is Diabetic or Non-Diabetic."
)

demo.launch(server_name="0.0.0.0", server_port=7860)