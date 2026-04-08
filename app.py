import streamlit as st
import numpy as np
import pickle
import gdown
import os
import re

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------
# CLEAN CSS FIX
# ---------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 36px;
    font-weight: 700;
    color: #4cc9f0;
}

.subtitle {
    color: #adb5bd;
    margin-bottom: 20px;
}

.card {
    background: #111827;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #1f2937;
}

.result-card {
    background: #0f766e;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}

.chat-user {
    background: #1f2937;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}

.chat-bot {
    background: #0f172a;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# MODEL LOADING
# ---------------------------
@st.cache_resource
def load_model():
    path = "disease_model.pkl"
    if not os.path.exists(path):
        url = "https://drive.google.com/uc?id=1-iHS567nLgAGSO-usLvSJ0sY65Fk1_G9"
        gdown.download(url, path, quiet=False)
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

# ---------------------------
# DISEASE INFO
# ---------------------------
disease_info = {
    "Diabetes": {
        "desc": "Chronic condition affecting blood sugar levels.",
        "precautions": ["Exercise", "Healthy diet", "Monitor sugar"]
    },
    "Heart Disease": {
        "desc": "Conditions affecting heart function.",
        "precautions": ["Avoid smoking", "Exercise", "Low salt diet"]
    },
    "Malaria": {
        "desc": "Mosquito-borne disease.",
        "precautions": ["Use nets", "Avoid stagnant water"]
    },
    "Typhoid": {
        "desc": "Food/water infection.",
        "precautions": ["Clean water", "Wash hands"]
    }
}

# ---------------------------
# HEADER
# ---------------------------
st.markdown('<div class="title">🩺 MediPredict AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart disease prediction using Machine Learning</div>', unsafe_allow_html=True)

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def extract_symptoms(text):
    text = text.lower()
    found = []
    for col in columns:
        if col.replace("_", " ") in text:
            found.append(col)
    return found

def predict(symptoms):
    vec = np.zeros(len(columns))
    for s in symptoms:
        if s in columns:
            vec[columns.index(s)] = 1
    return model.predict([vec])[0], vec

def show_result(pred, vec):
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.success(f"🩺 Predicted: {pred}")

    if pred in disease_info:
        st.write("📖", disease_info[pred]["desc"])
        st.write("🛡️ Precautions:")
        for p in disease_info[pred]["precautions"]:
            st.write("✔️", p)

    st.info("Consult a doctor for proper diagnosis.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# TABS
# ---------------------------
tab1, tab2 = st.tabs(["📋 Select Symptoms", "💬 Chat Assistant"])

# ---------------------------
# TAB 1: SELECT
# ---------------------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    selected = st.multiselect("Choose symptoms", columns)

    if st.button("🔍 Predict"):
        if selected:
            pred, vec = predict(selected)
            show_result(pred, vec)
        else:
            st.warning("Select symptoms first")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# TAB 2: CHAT
# ---------------------------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    user_input = st.text_area("Describe symptoms")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    if st.button("🧠 Analyze"):
        if user_input.strip():
            detected = extract_symptoms(user_input)

            st.session_state.chat.append(("user", user_input))

            if detected:
                st.session_state.chat.append(("bot", f"Detected: {', '.join(detected)}"))
                pred, vec = predict(detected)
                st.session_state.chat.append(("bot", f"Prediction: {pred}"))
            else:
                st.session_state.chat.append(("bot", "Could not detect symptoms"))

    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.write("⚠️ Educational use only. Not a medical diagnosis tool.")