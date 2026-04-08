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
# CLEAN CSS
# ---------------------------
st.markdown("""
<style>
div[data-testid="stVerticalBlock"] > div:empty { display: none !important; }
.block-container { padding-top: 1.2rem; max-width: 900px; }

.header {
    background: linear-gradient(90deg, #4cc9f0, #4361ee);
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.title { font-size: 30px; font-weight: 700; color: white; }
.subtitle { color: #e0e7ff; }

.card {
    background: #0f172a;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #1f2937;
    margin-top: 10px;
}

.result-card {
    background: #065f46;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
}

.chat-user {
    background: #2563eb;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: right;
}
.chat-bot {
    background: #1f2937;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
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
# HEADER
# ---------------------------
st.markdown("""
<div class="header">
    <div class="title">🩺 MediPredict AI</div>
    <div class="subtitle">AI-powered disease prediction</div>
</div>
""", unsafe_allow_html=True)

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

def predict_top3(symptoms):
    vec = np.zeros(len(columns))
    for s in symptoms:
        if s in columns:
            vec[columns.index(s)] = 1

    probs = model.predict_proba([vec])[0]
    classes = model.classes_

    top_idx = np.argsort(probs)[-3:][::-1]
    results = [(classes[i], probs[i]) for i in top_idx]

    return results, vec

# ---------------------------
# EXPLAIN PREDICTION
# ---------------------------
def explain_prediction(vec):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

        # get top contributing symptoms
        indices = np.argsort(importance)[-5:][::-1]

        st.subheader("🧠 Why this prediction?")
        st.write("Top contributing symptoms:")

        for i in indices:
            if vec[i] == 1:
                st.write(f"✔️ {columns[i]} (importance: {importance[i]:.2f})")

# ---------------------------
# SHOW RESULTS
# ---------------------------
def show_results(results, vec):
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.subheader("🩺 Top Predictions")

    for i, (disease, prob) in enumerate(results):
        pct = prob * 100
        if i == 0:
            st.success(f"🥇 {disease} ({pct:.2f}%)")
        elif i == 1:
            st.info(f"🥈 {disease} ({pct:.2f}%)")
        else:
            st.warning(f"🥉 {disease} ({pct:.2f}%)")

    explain_prediction(vec)

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
                results, vec = predict_top3(selected)
            show_results(results, vec)
        else:
            st.warning("Select symptoms first")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# TAB 2 (CHAT)
# ---------------------------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user_input = st.text_area("Describe your symptoms")

    if st.button("🧠 Analyze"):
        if user_input.strip():
            detected = extract_symptoms(user_input)

            st.session_state.chat.append(("user", user_input))

            with st.spinner("Thinking..."):
                time.sleep(1)

            if detected:
                results, vec = predict_top3(detected)

                st.session_state.chat.append(("bot", f"Detected: {', '.join(detected)}"))
                st.session_state.chat.append(("bot", f"Most likely: {results[0][0]}"))

                show_results(results, vec)
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
st.caption("⚠️ Educational use only. Not a medical diagnosis tool.")