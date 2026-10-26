from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "processed_student_data.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "deployment.pkl"
FEATURES_PATH = PROJECT_ROOT / "model" / "features.pkl"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded successfully")
X = df.drop(columns=["G3"])

X = pd.get_dummies(X)

features = joblib.load(FEATURES_PATH)
X = X.reindex(columns=features, fill_value=0)

model = joblib.load(MODEL_PATH)
print("Model Loaded successfully")

predictions = model.predict(X)

df["predicted_G3"] = predictions.round(2)
df.to_csv(PROJECT_ROOT / "batch_predictions.csv", index=False)
print("\nBatch Predictions completed")
print(df[["G3", "predicted_G3"]].head())
print("\nTotal students :", len(df))