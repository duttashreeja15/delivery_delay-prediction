
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent

# Sample features from your test dataset
X_test = pd.read_csv(
    BASE_DIR / "data" / "processed" / "X_test.csv"
)

# Use one existing order as test input
order = X_test.iloc[0].copy()

# Convert NumPy values to ordinary Python values
# and represent missing data as JSON null
payload = {
    "features": order.astype(object).where(order.notna(), None).to_dict()
}

try:
    response = requests.post(
        "http://127.0.0.1:8001/predict",
        json=payload,
        timeout=30
    )
except requests.exceptions.ConnectionError:
    raise SystemExit(
        "Cannot connect to the API at http://127.0.0.1:8001.\n"
        "Start it in another terminal from the project root:\n"
        ".venv/bin/python -m uvicorn backend.api:app --port 8001"
    ) from None
except requests.exceptions.Timeout:
    raise SystemExit("The API did not respond within 30 seconds.") from None

print("Status:", response.status_code)
try:
    print("Response:", response.json())
except requests.exceptions.JSONDecodeError:
    print("Response:", response.text)
response.raise_for_status()
