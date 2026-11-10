import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5001")

model_uri = (
    "models:/delivery_delay_lightgbm@champion"
)

model = mlflow.sklearn.load_model(model_uri)

print("Model loaded successfully!")
print("Model type:", type(model))