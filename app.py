import streamlit as st
import numpy as np
import pickle
import gdown
import os

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------
# CUSTOM CSS (UI UPGRADE)
# ---------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #2c3e50;
}
.subtitle {
    font-size: 18px;
    color: #7f8c8d;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}
.result-card {
    background: #e8f8f5;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #1abc9c;
}
.warning-card {
    background: #fff3cd;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# MODEL LOADING
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

model = load_model()

# Load columns
with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

# ---------------------------
# SIDEBAR (PRO LOOK)
# ---------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Adjust your input preferences")

confidence_toggle = st.sidebar.checkbox("Show Confidence Score", True)

st.sidebar.markdown("---")
st.sidebar.info("Developed for educational purposes")

# ---------------------------
# HEADER
# ---------------------------
st.markdown('<div class="title">🩺 MediPredict AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart disease prediction using Machine Learning</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# INPUT SECTION (CARD UI)
# ---------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🧾 Select Your Symptoms")

selected_symptoms = st.multiselect(
    "Choose symptoms",
    columns,
    help="Select all symptoms you are experiencing"
)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# PREPARE INPUT
# ---------------------------
input_data = np.zeros(len(columns))

for symptom in selected_symptoms:
    index = columns.index(symptom)
    input_data[index] = 1

# ---------------------------
# PREDICTION BUTTON
# ---------------------------
col1, col2 = st.columns([1, 3])

with col1:
    predict_btn = st.button("🔍 Predict")

# ---------------------------
# RESULT SECTION
# ---------------------------
if predict_btn:
    if len(selected_symptoms) == 0:
        st.markdown(
            '<div class="warning-card">⚠️ Please select at least one symptom</div>',
            unsafe_allow_html=True
        )
    else:
        prediction = model.predict([input_data])[0]

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        st.success(f"🩺 Predicted Disease: **{prediction}**")

        if confidence_toggle:
            try:
                probs = model.predict_proba([input_data])[0]
                confidence = max(probs) * 100
                st.info(f"📊 Confidence: {confidence:.2f}%")
            except:
                st.warning("Confidence not available")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")

st.markdown("""
<div style="text-align:center; color:gray;">
⚠️ This tool is for educational purposes only and not a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)