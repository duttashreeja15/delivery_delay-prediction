"""Train, validate, and register both delivery-delay pipelines in MLflow."""
from pathlib import Path
import os
import runpy

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from validate_model import main as validate_model

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    # Use one backend for training runs and the model registry.
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR / 'mlflow.db'}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(os.environ.get("MLFLOW_REGISTRY_URI", tracking_uri))
    print(f"MLflow tracking URI: {tracking_uri}")

    training = runpy.run_path(str(BASE_DIR / "python" / "train_new.py"), run_name="__main__")
    threshold = training["DECISION_THRESHOLD"]
    example = pd.read_csv(BASE_DIR / "data" / "processed" / "X_test.csv").head(5)

    # Validate all models before starting registration.
    validated = []
    for result in training["comparison"]:
        model, metrics = validate_model(result["model_path"], threshold)
        validated.append((result, model, metrics))

    for result, model, metrics in validated:
        registry_name = f"delivery_delay_{result['model']}"
        with mlflow.start_run(run_id=result["run_id"]):
            mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
            signature = infer_signature(example, model.predict_proba(example))
            # Serve probabilities so callers can apply the logged 0.3 threshold.
            info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="registered_pipeline",
                registered_model_name=registry_name,
                signature=signature,
                input_example=example,
                pyfunc_predict_fn="predict_proba",
                metadata={"decision_threshold": threshold, "positive_class": 1},
                pip_requirements=str(BASE_DIR / "requirements.txt"),
            )
            print(f"Registered {registry_name}: {info.model_uri}")

    print("Training, validation, and registration completed.")


if __name__ == "__main__":
    main()
