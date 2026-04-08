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
    layout="wide"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------
st.markdown("""
<style>
.main { background-color: #f5f7fa; }
.title { font-size: 40px; font-weight: bold; color: #2c3e50; }
.subtitle { font-size: 18px; color: #7f8c8d; }
.card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0px 4px 10px rgba(0,0,0,0.05); }
.result-card { background: #e8f8f5; padding: 20px; border-radius: 12px; border-left: 6px solid #1abc9c; }
.warning-card { background: #fff3cd; padding: 15px; border-radius: 10px; }
.chat-bubble { background:#ffffff; padding:12px 14px; border-radius:10px; margin:6px 0; box-shadow: 0px 2px 6px rgba(0,0,0,0.06); }
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
        return pickle.load(f)

model = load_model()

# Load feature names
with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

# ---------------------------
# DISEASE INFO DATABASE
# ---------------------------
disease_info = {
    "Diabetes": {
        "description": "A chronic condition that affects how your body processes blood sugar.",
        "precautions": [
            "Maintain a balanced diet",
            "Exercise regularly",
            "Monitor blood sugar levels",
            "Avoid excessive sugar intake"
        ]
    },
    "Heart Disease": {
        "description": "Conditions affecting heart function and blood vessels.",
        "precautions": [
            "Avoid smoking",
            "Maintain healthy weight",
            "Exercise regularly",
            "Reduce salt intake"
        ]
    },
    "Malaria": {
        "description": "A mosquito-borne infectious disease caused by parasites.",
        "precautions": [
            "Use mosquito nets",
            "Avoid stagnant water",
            "Wear full-sleeve clothes",
            "Use insect repellent"
        ]
    },
    "Typhoid": {
        "description": "A bacterial infection spread through contaminated food and water.",
        "precautions": [
            "Drink clean water",
            "Wash hands regularly",
            "Avoid street food",
            "Maintain hygiene"
        ]
    }
}

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("⚙️ Settings")
confidence_toggle = st.sidebar.checkbox("Show Confidence Score", True)
st.sidebar.markdown("---")
st.sidebar.info("Educational Project")

# ---------------------------
# HEADER
# ---------------------------
st.markdown('<div class="title">🩺 MediPredict AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart disease prediction using Machine Learning</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# HELPER: TEXT → SYMPTOMS
# ---------------------------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_symptoms_from_text(text, feature_list):
    text_n = normalize(text)
    found = []
    for feat in feature_list:
        # handle underscore or space variants
        variants = {feat.lower(), feat.lower().replace("_", " ")}
        for v in variants:
            if v in text_n:
                found.append(feat)
                break
    return list(set(found))

def build_input_vector(selected, feature_list):
    vec = np.zeros(len(feature_list))
    for s in selected:
        if s in feature_list:
            vec[feature_list.index(s)] = 1
    return vec

def render_result(prediction, input_vec):
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.success(f"🩺 Predicted Disease: **{prediction}**")

    if confidence_toggle:
        try:
            probs = model.predict_proba([input_vec])[0]
            confidence = max(probs) * 100
            st.info(f"📊 Confidence: {confidence:.2f}%")
        except:
            st.warning("Confidence not available")

    if prediction in disease_info:
        info = disease_info[prediction]
        st.subheader("📖 About the Disease")
        st.write(info["description"])
        st.subheader("🛡️ Precautions")
        for p in info["precautions"]:
            st.write(f"✔️ {p}")
    else:
        st.warning("No additional info available")

    st.info("👨‍⚕️ Please consult a doctor for proper diagnosis.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# TABS: SELECT / CHATBOT
# ---------------------------
tab_select, tab_chat = st.tabs(["🧾 Select Symptoms", "💬 Chatbot Input"])

# ---------------------------
# TAB 1: MULTISELECT (existing)
# ---------------------------
with tab_select:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Choose your symptoms")
    selected_symptoms = st.multiselect("Symptoms", columns)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Predict (Select)"):
        if not selected_symptoms:
            st.markdown('<div class="warning-card">⚠️ Please select at least one symptom</div>', unsafe_allow_html=True)
        else:
            vec = build_input_vector(selected_symptoms, columns)
            pred = model.predict([vec])[0]
            render_result(pred, vec)

# ---------------------------
# TAB 2: CHATBOT INPUT
# ---------------------------
with tab_chat:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Describe your symptoms in natural language")
    st.caption("Example: I have fever, headache and vomiting since yesterday")

    user_text = st.text_area("Type your symptoms here")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🧠 Analyze Symptoms"):
        if not user_text.strip():
            st.warning("Please enter your symptoms.")
        else:
            extracted = extract_symptoms_from_text(user_text, columns)

            st.session_state.chat_history.append(("user", user_text))
            if extracted:
                st.session_state.chat_history.append(("bot", f"Detected symptoms: {', '.join(extracted)}"))
                vec = build_input_vector(extracted, columns)
                pred = model.predict([vec])[0]
                st.session_state.chat_history.append(("bot", f"Predicted Disease: {pred}"))
            else:
                st.session_state.chat_history.append(("bot", "I couldn't match symptoms. Try using clearer keywords."))

    # Chat display
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f'<div class="chat-bubble"><b>🧑 You:</b> {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble"><b>🤖 MediPredict:</b> {msg}</div>', unsafe_allow_html=True)

    # Show last prediction details (if any)
    if st.session_state.chat_history:
        last_bot_msgs = [m for r, m in st.session_state.chat_history if r == "bot"]
        # If a prediction message exists, recompute vector for display details
        if last_bot_msgs and last_bot_msgs[-1].startswith("Predicted Disease:"):
            extracted = extract_symptoms_from_text(user_text, columns)
            if extracted:
                vec = build_input_vector(extracted, columns)
                pred = model.predict([vec])[0]
                render_result(pred, vec)

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