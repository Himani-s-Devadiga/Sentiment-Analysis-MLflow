import mlflow.sklearn


# Replace with your latest Run ID

model_uri = (
    "runs:/YOUR_RUN_ID/sentiment_model"
)


model = mlflow.sklearn.load_model(
    model_uri
)



reviews = [
    "This product is amazing, I love it",
    "Worst product ever, very disappointed"
]


predictions = model.predict(
    reviews
)


for review, prediction in zip(
        reviews,
        predictions):

    print("--------------------")
    print("Review:", review)
    print("Sentiment:", prediction)