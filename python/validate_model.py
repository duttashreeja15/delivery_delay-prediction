import pickle
import pandas as pd

with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")
print("Model type:", type(model))

# Use an actual unseen row with the exact
# features and preprocessing expected by the model.
X_test = pd.read_csv("data/processed/X_test.csv")

predictions = model.predict(X_test)

print("Predictions:", predictions)

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X_test.head(5))[:, 1]
    print("Delay probabilities:", probabilities)