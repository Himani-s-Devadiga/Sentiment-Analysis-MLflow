import streamlit as st
import mlflow.sklearn
import joblib

from utils import clean_text

# Page configuration (must come first)
st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="🛍️",
    layout="centered"
)

# Load model
model = mlflow.sklearn.load_model("sentiment-model")

# Load TF-IDF vectorizer
tfidf = joblib.load("tfidf.pkl")

# Title
st.title("🛍️ Amazon Review Sentiment Analysis")

st.write(
    "Enter an Amazon product review below and the model will predict whether it is Positive or Negative."
)

# Input
review = st.text_area("Enter Review")

# Prediction
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        cleaned_review = clean_text(review)

        vector = tfidf.transform([cleaned_review])

        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("😊 Positive Review")

        else:
            st.error("😞 Negative Review")