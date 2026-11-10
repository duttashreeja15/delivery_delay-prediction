
from pathlib import Path
import os

import pandas as pd
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8001"
).rstrip("/")

DATA_PATH = (
    BASE_DIR / "data" / "processed" / "X_test.csv"
)

st.set_page_config(
    page_title="Delivery Delay Prediction",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Delivery Delay Prediction System")

st.write(
    "Predict whether an e-commerce order may "
    "arrive after its estimated delivery date."
)

st.caption(
    "Machine Learning Portfolio Project | "
    "Olist Brazilian E-Commerce Dataset"
)


@st.cache_data
def load_examples():
    X_test = pd.read_csv(DATA_PATH)

    y_test = pd.read_csv(
        BASE_DIR / "data" / "processed" / "y_test.csv"
    )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Features and labels have different row counts"
        )

    return X_test, y_test


try:
    orders, labels = load_examples()
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load example orders: {error}")
    st.stop()

if orders.empty:
    st.warning("No example orders are available.")
    st.stop()

st.subheader("Select an example order")
order_index = st.selectbox(
    "Order",
    options=range(min(len(orders), 100)),
    format_func=lambda i: f"Example Order {i + 1}"
)
selected_order = orders.iloc[[order_index]]

st.subheader("Order information")
st.dataframe(selected_order.T.astype(str), width="stretch")

if st.button("Predict Delay Risk", type="primary"):
    order = selected_order.iloc[0]
    payload = {
        "features": order.astype(object).where(order.notna(), None).to_dict()
    }

    try:
        response = requests.post(
            f"{API_URL}/predict", json=payload, timeout=30
        )
        response.raise_for_status()
        result = response.json()
        score = float(result["delivery_delay_risk_score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError("Risk score must be between 0 and 1.")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to the prediction API at {API_URL}.")
        st.info("Start the backend from the project folder in another terminal:")
        st.code(
            ".venv/bin/python -m uvicorn backend.api:app --port 8001",
            language="bash"
        )
    except requests.exceptions.Timeout:
        st.error("The prediction API did not respond within 30 seconds.")
    except requests.RequestException as error:
        st.error(f"API request failed: {error}")
    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Invalid prediction response: {error}")
    
    else:
        st.subheader("Prediction Results")

        # Retrieve the actual outcome for this order
        actual = int(
            labels.iloc[order_index]["Delivery_delay"]
        )

        # Use the threshold selected during model validation
        threshold = 0.5

        predicted = int(score >= threshold)

        # Display three metrics side by side
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Delivery Delay Risk Score",
                f"{score:.1%}"
            )

        with col2:
            st.metric(
                "Predicted Outcome",
                "Delayed" if predicted == 1 else "On Time"
            )

        with col3:
            st.metric(
                "Actual Outcome",
                "Delayed" if actual == 1 else "On Time"
            )

        # Visualize the risk score
        st.progress(score)

        st.caption(
            f"Classification threshold: {threshold:.2f}. "
            "This score is not necessarily a calibrated "
            "probability."
        )

        # Evaluate the prediction
        st.subheader("Prediction Evaluation")

        if predicted == 1 and actual == 1:
            st.success(
                "True Positive (TP): "
                "Correctly predicted a delayed order."
            )

        elif predicted == 1 and actual == 0:
            st.warning(
                "False Positive (FP): "
                "Predicted a delay, but the order "
                "was delivered on time."
            )

        elif predicted == 0 and actual == 1:
            st.error(
                "False Negative (FN): "
                "The model missed an actual delivery delay."
            )

        else:
            st.success(
                "True Negative (TN): "
                "Correctly predicted an on-time delivery."
            )

