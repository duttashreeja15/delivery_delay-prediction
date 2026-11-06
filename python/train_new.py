
from pathlib import Path
import pickle

import pandas as pd
import mlflow

from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.3
TEST_SIZE = 0.30

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "merged_orders.csv"
MODEL_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "delivery_delay_pipeline.pkl"

NUMERICAL_COLUMNS = [
    "price",
    "freight_value",
    "payment_installments",
    "payment_value",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "purchase_month",
    "purchase_dayofweek",
    "purchase_hour",
    "number_of_items",
    "unique_products",
    "number_of_sellers",
    "number_of_payments",
    "total_product_weight_kg",
    "avg_product_weight_kg",
    "total_product_volume_cm3",
    "shipping_distance_km",
]

CATEGORICAL_COLUMNS = [
    "seller_state",
    "customer_state",
    "payment_type",
    "product_category_name",
]

FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

date_columns = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

required_columns = (
    {"order_id", "order_status"}
    | set(date_columns)
    | (set(FEATURE_COLUMNS) - {"purchase_month", "purchase_dayofweek", "purchase_hour"})
)
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(
        f"Missing query result columns: {sorted(missing_columns)}. "
        "Regenerate data/merged_orders.csv using query/query.sql."
    )

if df["order_id"].isna().any() or df["order_id"].duplicated().any():
    raise ValueError("Expected one non-null, unique order_id per row from query/query.sql.")

for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors="coerce")

print(f"Original rows: {len(df):,}")


# --------------------------------------------------
# 3. Filter eligible orders
# --------------------------------------------------

# Initial scope: completed, delivered orders only.
# Non-delivered orders have an unknown delivery outcome.

df = df[
    (df["order_status"] == "delivered")
    & df["order_purchase_timestamp"].notna()
    & df["order_delivered_customer_date"].notna()
    & df["order_estimated_delivery_date"].notna()
    & df["order_id"].notna()
].copy()

# The SQL export already aggregates items and payments to one row per order.

print(f"Eligible unique orders: {len(df):,}")

if df.empty:
    raise ValueError("No eligible orders found.")


# --------------------------------------------------
# 4. Create target
# --------------------------------------------------

# 1 = delivered after promised calendar date
# 0 = delivered on or before promised calendar date

df["Delivery_delay"] = (
    df["order_delivered_customer_date"].dt.normalize()
    > df["order_estimated_delivery_date"].dt.normalize()
).astype(int)

print(
    "Late delivery rate:",
    round(df["Delivery_delay"].mean() * 100, 2),
    "%"
)


# --------------------------------------------------
# 5. Feature engineering
# --------------------------------------------------

# Use the distance calculated from the aggregated coordinates in the SQL export.
for column in NUMERICAL_COLUMNS:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
df.loc[df["shipping_distance_km"] < 0, "shipping_distance_km"] = float("nan")
print("Orders without shipping distance:", df["shipping_distance_km"].isna().sum())

df["purchase_month"] = (
    df["order_purchase_timestamp"].dt.month
)

df["purchase_dayofweek"] = (
    df["order_purchase_timestamp"].dt.dayofweek
)

df["purchase_hour"] = (
    df["order_purchase_timestamp"].dt.hour
)

df["estimated_delivery_days"] = (
    (
        df["order_estimated_delivery_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
)

# Exclude impossible or inconsistent promised dates.
df = df[df["estimated_delivery_days"] >= 0].copy()

# Make sure all expected columns exist.
missing_columns = set(FEATURE_COLUMNS) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing feature columns: {missing_columns}"
    )

X = df[FEATURE_COLUMNS].copy()
y = df["Delivery_delay"].copy()

print(df["Delivery_delay"].value_counts())

print(
    df["Delivery_delay"].value_counts(normalize=True) * 100
)

# --------------------------------------------------
# 6. Chronological train/test split
# --------------------------------------------------

df = df.sort_values(
    ["order_purchase_timestamp", "order_id"]
).reset_index(drop=True)

X = df[FEATURE_COLUMNS]
y = df["Delivery_delay"]

split_index = int(len(df) * (1 - TEST_SIZE))

if split_index <= 0 or split_index >= len(df):
    raise ValueError("Not enough rows to create split.")

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Both train and test sets must contain both classes."
    )

print(f"Train orders: {len(X_train):,}")
print(f"Test orders: {len(X_test):,}")


print(
    "Test begins:",
    df.iloc[split_index]["order_purchase_timestamp"]
)


# --------------------------------------------------
# 7. Preprocessing pipeline
# --------------------------------------------------

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    )),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, NUMERICAL_COLUMNS),
        ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
    ]
)


# --------------------------------------------------
# 8. ML model pipeline
# --------------------------------------------------

models = {
    "random_forest": RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=100, learning_rate=0.05, num_leaves=31,
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
        verbosity=-1,
    ),
}

# Save the same holdout for both models.
X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
mlflow.set_experiment("Delivery_Delay_Prediction")
comparison = []

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", model),
    ])
    model_path = (
        MODEL_PATH if model_name == "random_forest"
        else MODEL_DIR / "delivery_delay_lightgbm_pipeline.pkl"
    )
    with mlflow.start_run(run_name=f"{model_name}_pipeline"):
        mlflow.log_params(model.get_params())
        mlflow.log_params({
            "model": type(model).__name__,
            "test_size": TEST_SIZE,
            "split_method": "chronological",
            "decision_threshold": DECISION_THRESHOLD,
        })
        # Train once
        pipeline.fit(X_train, y_train)
            

        y_proba = pipeline.predict_proba(X_test)[:, 1]

        print(pd.Series(y_proba).describe())

        print("Maximum probability:", y_proba.max())
        print("Predicted positive cases:", sum(y_proba >= DECISION_THRESHOLD))

        # Predictions and probabilities
        y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "test_recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "test_f1": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "test_pr_auc": average_precision_score(
                y_test, y_proba
            ),
        }

        # How many delays are captured in the
        # riskiest 10% of test orders?
        top_n = max(1, int(len(y_proba) * 0.10))
        top_indices = y_proba.argsort()[-top_n:]

        captured_delays = y_test.iloc[top_indices].sum()
        total_delays = y_test.sum()

        metrics["recall_at_top_10pct"] = (
            float(captured_delays / total_delays)
            if total_delays > 0 else 0.0
        )

        mlflow.log_metrics(metrics)
        comparison.append({"model": model_name, **metrics})

        print("\nModel Evaluation")
        print("----------------------")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nClassification report:")
        print(classification_report(
            y_test, y_pred, zero_division=0
        ))

        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Save complete preprocessing + classifier pipeline
        with model_path.open("wb") as file:
            pickle.dump(pipeline, file)

        # Store model artifact in MLflow
        mlflow.log_artifact(
            str(model_path),
            artifact_path="model"
        )

        # Save untouched test features and labels
        X_test.to_csv(
            PROCESSED_DIR / "X_test.csv",
            index=False
        )

        y_test.to_csv(
            PROCESSED_DIR / "y_test.csv",
            index=False
        )

        print("\nModel saved to:", model_path)
        print("MLflow run ID:", mlflow.active_run().info.run_id)

    print("Training completed successfully!")

    #Diagnostics
    print("Training target distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTest target distribution:")
    print(y_test.value_counts(normalize=True))

    print("\nPrediction distribution:")
    print(pd.Series(y_pred).value_counts())

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nProbability statistics:")
    print(pd.Series(y_proba).describe())


    monthly = (
        df.assign(
            month=df["order_purchase_timestamp"]
            .dt.to_period("M")
        )
        .groupby("month")["Delivery_delay"]
        .agg(["mean", "count"])
    )

    print(monthly)


    import pandas as pd

    analysis = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted_probability": y_proba
    })

    analysis["risk_group"] = pd.qcut(
        analysis["predicted_probability"],
        q=10,
        duplicates="drop"
    )

    risk_summary = (
        analysis.groupby(
            "risk_group",
            observed=True
        )
        .agg(
            orders=("actual", "count"),
            actual_delay_rate=("actual", "mean"),
            avg_predicted_probability=(
                "predicted_probability", "mean"
            )
        )
    )

    print(risk_summary)

print("\nModel comparison (threshold =", DECISION_THRESHOLD, ")")
print(pd.DataFrame(comparison).set_index("model").to_string())
