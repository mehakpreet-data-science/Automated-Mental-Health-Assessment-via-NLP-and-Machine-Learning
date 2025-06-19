
import streamlit as st
import joblib
import re
import string

# Load the model and vectorizer
model = joblib.load("mental_health_model.pkl")
vectorizer = joblib.load("mental_health_vec.pkl")

# Define text cleaning function (simple version)
def clean_text(text):
    text = text.lower()                      # Lowercase
    text = re.sub(r"\d+", "", text)          # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()                      # Trim whitespace
    return text

# Streamlit UI
st.title("🧠 Mental Health Text Classifier")
st.markdown("This NLP app classifies mental health-related statements into categories (e.g., Stress, Anxiety, etc.).")

# Text input
user_input = st.text_area("Enter your mental health-related statement:", height=150)

# Prediction
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a statement.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"✅ **Predicted Mental Health Category:** {prediction}")
