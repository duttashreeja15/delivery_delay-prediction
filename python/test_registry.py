import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5001")

MODEL_URI = "models:/delivery_delay_lightgbm@champion"

model = mlflow.sklearn.load_model(MODEL_URI)

print("Champion model loaded successfully!")
print("Model type:", type(model))

if hasattr(model, "predict_proba"):
    print("predict_proba is available")