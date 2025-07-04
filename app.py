
import streamlit as st
import joblib

# Load models and tools
lr_model = joblib.load("logistic_model.pkl")
nb_model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit UI
st.title("📊 Sentiment Analysis on Reviews")
st.markdown("Enter your product review below. The app will predict the **sentiment** of the review.")

# Text input
review_text = st.text_area("📝 Enter your review text:")

# Model selection
model_choice = st.selectbox("Choose model:", ["Logistic Regression", "Naive Bayes"])

# Prediction
if st.button("🔍 Predict"):
    if not review_text.strip():
        st.warning("⚠️ Please enter review text.")
    else:
        vect_input = vectorizer.transform([review_text])
        model = lr_model if model_choice == "Logistic Regression" else nb_model
        prediction = model.predict(vect_input)[0]
        sentiment = label_encoder.inverse_transform([prediction])[0]

        # Add emoji based on sentiment
        emoji = {
            "Positive": "😊👍",
            "Neutral": "😐",
            "Negative": "😠👎"
        }.get(sentiment, "")

        st.success(f"🧠 **Predicted Sentiment:** {sentiment} {emoji}")
