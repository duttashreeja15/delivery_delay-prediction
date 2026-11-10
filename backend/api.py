
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{BASE_DIR / 'mlflow.db'}"
)

MODEL_URI = "models:/delivery_delay_lightgbm@champion"

mlflow.set_tracking_uri(TRACKING_URI)

# Load model once when the API starts
model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(
    title="Delivery Delay Prediction API",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    features: dict


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_URI
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert request to one-row DataFrame
        input_df = pd.DataFrame([request.features])

        # Align input features to the fitted pipeline
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            missing = [
                col for col in expected
                if col not in input_df.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing features: {missing}"
                )

            input_df = input_df[expected]

        # Get model risk score
        risk_score = float(
            model.predict_proba(input_df)[0, 1]
        )

        return {
            "delivery_delay_risk_score": risk_score,
            "model": MODEL_URI
        }

    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )
