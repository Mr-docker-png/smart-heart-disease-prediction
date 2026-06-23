import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================
# Load Model
# ==========================

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "heart_model.pkl")

# ==========================
# Page Settings
# ==========================

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# ==========================
# Title
# ==========================

st.title("❤️ Heart Disease Risk Predictor")

st.write(
    "Enter your health information below to estimate heart disease risk."
)

# ==========================
# User Inputs
# ==========================

age = st.slider(
    "Age",
    20,
    100,
    45
)

sex = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

cp = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ]
)

trestbps = st.slider(
    "Resting Blood Pressure",
    80,
    220,
    120
)

chol = st.slider(
    "Cholesterol",
    100,
    600,
    200
)

thalach = st.slider(
    "Maximum Heart Rate",
    60,
    220,
    150
)

# ==========================
# Encoding Inputs
# ==========================

sex_value = 1 if sex == "Male" else 0

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}

cp_value = cp_map[cp]

# ==========================
# Default Values
# Hidden Inputs
# ==========================

fbs = 0
restecg = 1
exang = 0
oldpeak = 1.0
slope = 1
ca = 0
thal = 2

# ==========================
# Prediction
# ==========================

if st.button("🔍 Check Risk"):

    data = pd.DataFrame(
        [[
            age,
            sex_value,
            cp_value,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]],
        columns=[
            'age',
            'sex',
            'cp',
            'trestbps',
            'chol',
            'fbs',
            'restecg',
            'thalach',
            'exang',
            'oldpeak',
            'slope',
            'ca',
            'thal'
        ]
    )

    prediction = model.predict(data)[0]

    st.markdown("---")

    if prediction == 1:

        st.error(
            "⚠️ High Risk of Heart Disease"
        )

        st.write(
            "Please consult a healthcare professional for proper medical advice."
        )

    else:

        st.success(
            "❤️ Low Risk of Heart Disease"
        )

        st.write(
            "Your prediction indicates a lower risk based on the provided information."
        )

# ==========================
# Footer
# ==========================

st.markdown("---")

st.markdown(
    """
    ### 🛠 Built With

    - Python
    - Streamlit
    - Scikit-Learn
    - Decision Tree Classifier
    """
)