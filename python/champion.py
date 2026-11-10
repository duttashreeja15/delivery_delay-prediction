import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5001")

client = MlflowClient()

client.set_registered_model_alias(
    name="delivery_delay_lightgbm",
    alias="champion",
    version=1
)