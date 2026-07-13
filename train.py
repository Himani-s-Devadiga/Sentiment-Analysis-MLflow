import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from utils import clean_text, create_sentiment


# -----------------------------
# Load Dataset
# -----------------------------

print("Loading dataset...")

df = pd.read_csv("data.csv")

print("Dataset Loaded")
print(df.head())
print(df.shape)


# -----------------------------
# Remove Missing Values
# -----------------------------

print("\nRemoving missing values...")

df = df.dropna()

print("After removing missing values:")
print(df.shape)


# -----------------------------
# Create Sentiment Label
# -----------------------------

print("\nCreating sentiment labels...")

df = create_sentiment(df)

print(df.head())


# -----------------------------
# Convert Sentiment to Numeric
# -----------------------------

print("\nConverting sentiment labels...")

df["Sentiment"] = df["Sentiment"].map({
    "Negative": 0,
    "Positive": 1
})


df = df.dropna(subset=["Sentiment"])

df["Sentiment"] = df["Sentiment"].astype(int)


# -----------------------------
# Text Cleaning
# -----------------------------

print("\nCleaning text...")

df["clean_text"] = df["Review Text"].apply(clean_text)

print("Text cleaning completed!")


# -----------------------------
# Train Test Split
# -----------------------------

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["Sentiment"],
    test_size=0.2,
    random_state=42
)


# -----------------------------
# TF-IDF Vectorization
# -----------------------------

print("\nApplying TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF completed!")


# Save vectorizer

joblib.dump(
    tfidf,
    "tfidf.pkl"
)

print("TF-IDF vectorizer saved!")


# -----------------------------
# Model Training
# -----------------------------

print("\nTraining model...")

model = LogisticRegression(
    max_iter=1000
)


# -----------------------------
# MLflow Tracking
# -----------------------------

mlflow.set_experiment(
    "Amazon Review Sentiment Analysis"
)


with mlflow.start_run():

    model.fit(
        X_train_tfidf,
        y_train
    )

    print("Model training completed!")


    # Prediction

    y_pred = model.predict(
        X_test_tfidf
    )


    # Evaluation

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    print("\nAccuracy:", accuracy)


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )


    # Log metric

    mlflow.log_metric(
        "accuracy",
        accuracy
    )


    # Register model

    mlflow.sklearn.log_model(
        sk_model=model,
        name="sentiment-model",
        registered_model_name="SentimentAnalysisModel"
    )


print("\nTraining Completed Successfully!")