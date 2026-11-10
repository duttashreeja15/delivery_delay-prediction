
from pathlib import Path
import os
import shutil

import mlflow
import mlflow.artifacts
from mlflow import MlflowClient

BASE_DIR = Path(__file__).resolve().parent.parent

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{BASE_DIR / 'mlflow.db'}"
)
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", TRACKING_URI))

MODEL_NAME = "delivery_delay_lightgbm"
MODEL_ALIAS = "champion"

client = MlflowClient()

print(f"MLflow tracking URI: {TRACKING_URI}", flush=True)
print(f"Resolving {MODEL_NAME}@{MODEL_ALIAS}...", flush=True)
version = client.get_model_version_by_alias(
    MODEL_NAME, MODEL_ALIAS
)

print("Exporting model version:", version.version, flush=True)

model_uri = f"models:/{MODEL_NAME}/{version.version}"

print("Downloading model artifacts...", flush=True)
downloaded_path = Path(
    mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri
    )
)

export_path = BASE_DIR / "deployment" / "model"

export_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

if export_path.exists():
    shutil.rmtree(export_path)

shutil.copytree(downloaded_path, export_path)

print("Model exported to:", export_path)
print("Model version:", version.version)
