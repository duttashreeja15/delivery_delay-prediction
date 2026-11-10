"""Evaluate the saved training pipeline on its exported holdout data."""
from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, classification_report, confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "delivery_delay_pipeline.pkl"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DECISION_THRESHOLD = 0.3


def main(model_path=MODEL_PATH, threshold=DECISION_THRESHOLD):
    model_path = Path(model_path)
    paths = [model_path, PROCESSED_DIR / "X_test.csv", PROCESSED_DIR / "y_test.csv"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing validation files: {missing}. Run python/train_new.py first."
        )

    with model_path.open("rb") as file:
        model = pickle.load(file)
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    labels = pd.read_csv(PROCESSED_DIR / "y_test.csv")
    if list(labels.columns) != ["Delivery_delay"]:
        raise ValueError("Expected y_test.csv with one Delivery_delay column. Rerun train_new.py.")
    y_test = labels["Delivery_delay"]
    if X_test.empty or len(X_test) != len(y_test):
        raise ValueError("Test features and labels must have the same nonzero row count.")
    if not y_test.isin([0, 1]).all() or y_test.nunique() != 2:
        raise ValueError("Test labels must contain both binary classes, 0 and 1.")

    expected = list(model.feature_names_in_)
    if "shipping_distance_km" not in expected:
        raise ValueError("Saved model uses the old features. Rerun python/train_new.py.")
    missing_features = set(expected) - set(X_test.columns)
    extra_features = set(X_test.columns) - set(expected)
    if missing_features or extra_features:
        raise ValueError(
            f"Test features do not match the saved model. Missing: {sorted(missing_features)}; "
            f"unexpected: {sorted(extra_features)}. Rerun python/train_new.py."
        )
    X_test = X_test[expected]
    positive_class = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, positive_class]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_precision": precision_score(y_test, predictions, zero_division=0),
        "test_recall": recall_score(y_test, predictions, zero_division=0),
        "test_f1": f1_score(y_test, predictions, zero_division=0),
        "test_pr_auc": average_precision_score(y_test, probabilities),
    }
    top_n = max(1, int(len(probabilities) * 0.10))
    metrics["recall_at_top_10pct"] = float(
        y_test.iloc[probabilities.argsort()[-top_n:]].sum() / y_test.sum()
    )
    print(f"Model: {model_path}")
    print(f"Test orders: {len(X_test):,}; threshold: {threshold}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (rows=actual, columns=predicted; classes=[0, 1]):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nSample predictions:")
    print(pd.DataFrame({
        "actual": y_test,
        "delay_probability": probabilities,
        "predicted": predictions,
    }).head(10).to_string(index=False))

    return model, metrics


if __name__ == "__main__":
    main()
