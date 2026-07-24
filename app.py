import gradio as gr
import joblib
import numpy as np
import matplotlib.pyplot as plt

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
    # Input data
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

    # Scale data
    data = scaler.transform(data)

    # Prediction
    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "🩺 Diabetic"
    else:
        result = "✅ Non-Diabetic"

    # ------------------------
    # Visualization
    # ------------------------

    features = [
        "Preg",
        "Glucose",
        "BP",
        "Skin",
        "Insulin",
        "BMI",
        "DPF",
        "Age"
    ]

    values = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]

    fig, ax = plt.subplots(figsize=(9, 4))

    bars = ax.bar(features, values)

    ax.set_title("Patient Parameters")
    ax.set_ylabel("Value")

    # Values on bars
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h,
            f"{h}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.xticks(rotation=45)
    plt.tight_layout()

    return result, fig


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
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Plot(label="Visualization"),
    ],
    title="🩺 Diabetes Prediction System",
    description="Enter patient details to predict whether the patient is Diabetic or Non-Diabetic.",
)

demo.launch(inbrowser=True)
