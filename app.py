import streamlit as st
import numpy as np
import pickle

# Load model
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature names
with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

st.set_page_config(page_title="Disease Prediction System", layout="centered")

st.title("🏥 Disease Prediction System")
st.write("Select symptoms and get predicted disease")

# Multi-select symptoms
selected_symptoms = st.multiselect("Select Symptoms", columns)

# Create input vector
input_data = np.zeros(len(columns))

for symptom in selected_symptoms:
    index = columns.index(symptom)
    input_data[index] = 1

# Prediction
if st.button("Predict Disease"):
    prediction = model.predict([input_data])[0]

    st.success(f"Predicted Disease: {prediction}")

    # If model supports probabilities
    try:
        probs = model.predict_proba([input_data])[0]
        confidence = max(probs) * 100
        st.info(f"Confidence: {confidence:.2f}%")
    except:
        st.warning("Probability not available for this model")

# Optional: Info Section
st.markdown("---")
st.subheader("⚠️ Disclaimer")
st.write("This system is for educational purposes only and not a medical diagnosis tool.")