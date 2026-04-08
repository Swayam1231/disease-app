import streamlit as st
import numpy as np
import pickle
import gdown
import os

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Disease Prediction System", layout="centered")

# ---------------------------
# DOWNLOAD MODEL FUNCTION
# ---------------------------
@st.cache_resource
def load_model():
    model_path = "disease_model.pkl"

    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1-iHS567nLgAGSO-usLvSJ0sY65Fk1_G9"
        gdown.download(url, model_path, quiet=False)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

# ---------------------------
# LOAD DATA
# ---------------------------
model = load_model()

# Load feature names
with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

# ---------------------------
# UI
# ---------------------------
st.title("🏥 Disease Prediction System")
st.markdown("Select your symptoms and get a predicted disease")

# Multi-select symptoms
selected_symptoms = st.multiselect("Select Symptoms", columns)

# Create input vector
input_data = np.zeros(len(columns))

for symptom in selected_symptoms:
    index = columns.index(symptom)
    input_data[index] = 1

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("⚠️ Please select at least one symptom")
    else:
        prediction = model.predict([input_data])[0]

        st.success(f"🩺 Predicted Disease: {prediction}")

        # Confidence (if available)
        try:
            probs = model.predict_proba([input_data])[0]
            confidence = max(probs) * 100
            st.info(f"📊 Confidence: {confidence:.2f}%")
        except:
            st.warning("Confidence score not available")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.subheader("⚠️ Disclaimer")
st.write(
    "This system is for educational purposes only and should not be used as a medical diagnosis tool."
)