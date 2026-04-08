import streamlit as st
import numpy as np
import pickle
import gdown
import os
import time

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="MediPredict AI", page_icon="🩺", layout="centered")

# ---------------------------
# ADVANCED UI CSS
# ---------------------------
st.markdown("""
<style>

/* REMOVE EMPTY BLOCKS */
div[data-testid="stVerticalBlock"] > div:empty {
    display: none !important;
}

/* GLOBAL */
.block-container {
    padding-top: 1.2rem;
    max-width: 900px;
}

/* HEADER */
.header {
    background: linear-gradient(90deg, #4cc9f0, #4361ee);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.title {
    font-size: 32px;
    font-weight: 700;
    color: white;
}
.subtitle {
    color: #e0e7ff;
}

/* CARD */
.card {
    background: #0f172a;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #1f2937;
    margin-top: 10px;
}

/* RESULT */
.result-card {
    background: #065f46;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
}

/* CHAT */
.chat-user {
    background: #2563eb;
    color: white;
    padding: 10px 14px;
    border-radius: 12px;
    margin: 6px 0;
    text-align: right;
}
.chat-bot {
    background: #1f2937;
    padding: 10px 14px;
    border-radius: 12px;
    margin: 6px 0;
}

/* BUTTON */
.stButton button {
    border-radius: 10px;
    height: 45px;
}

/* INPUT FIX */
.stTextArea textarea {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL
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
# HEADER UI
# ---------------------------
st.markdown("""
<div class="header">
    <div class="title">🩺 MediPredict AI</div>
    <div class="subtitle">Smart disease prediction using Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# HELPERS
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
    return model.predict([vec])[0]

def show_result(pred):
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.success(f"🩺 Predicted Disease: {pred}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# TABS
# ---------------------------
tab1, tab2 = st.tabs(["📋 Select Symptoms", "💬 Chat Assistant"])

# ---------------------------
# TAB 1
# ---------------------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    selected = st.multiselect("Choose symptoms", columns)

    if st.button("🔍 Predict"):
        if selected:
            with st.spinner("Analyzing..."):
                time.sleep(1)
                pred = predict(selected)
            show_result(pred)
        else:
            st.warning("Select symptoms first")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# TAB 2 (CHAT UI)
# ---------------------------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user_input = st.text_area("Describe your symptoms", placeholder="I have fever and headache")

    if st.button("🧠 Analyze"):
        if user_input.strip():
            detected = extract_symptoms(user_input)

            st.session_state.chat.append(("user", user_input))

            with st.spinner("Thinking..."):
                time.sleep(1)

            if detected:
                pred = predict(detected)
                st.session_state.chat.append(("bot", f"Detected: {', '.join(detected)}"))
                st.session_state.chat.append(("bot", f"Prediction: {pred}"))
            else:
                st.session_state.chat.append(("bot", "Could not detect symptoms"))

    # CHAT DISPLAY
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
st.caption("⚠️ Educational use only. Not a medical diagnosis tool.")