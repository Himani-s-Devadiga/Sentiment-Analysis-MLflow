import mlflow
import mlflow.sklearn

model_uri = "runs:/167a82facb2343ad9e7dd0018fda6bec/sentiment-model"

model = mlflow.sklearn.load_model(model_uri)

mlflow.sklearn.save_model(
    sk_model=model,
    path="sentiment-model"
)

print("Model downloaded successfully!")