from pathlib import Path

from flask import Flask, jsonify
import joblib
import pandas as pd
import logging

BASE_DIR = Path(__file__).resolve().parent

# ------------------------------
# Configure Logging
# ------------------------------

logging.basicConfig(
    filename="prediction_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# create Flask App

app = Flask(__name__)
logging.info("Student Performance API Started")

# Load Model

model = joblib.load(BASE_DIR / "model" / "deployment.pkl")

# Health Check API

@app.route("/health", methods=["GET"])
def health():
    logging.info("Health API Accessed")
    return jsonify({
        "status" : "API is Running successfully"
    })

# Prediction API
@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Load the Dataset
        df = pd.read_csv(BASE_DIR / "processed_student_data.csv")
        X=df.drop(columns=["G3"])
        # Match Feature Names
        expected_columns = model.feature_names_in_

        #Add missing columns
        for column in expected_columns:
            if column not in X.columns:
                X[column] = 0

        # Keep only expected columns in correct order
        X = X[expected_columns]

        # Predict First Student
        prediction = model.predict(X.iloc[[0]])
        actual = int(df.iloc[0]["G3"])
        logging.info(
            f"Prediction Requested | Actual {actual} | Predicted : {round(float(prediction[0]), 2)}"
        )
        return jsonify({
            "Predicted_G3": round(float(prediction[0]), 2),
            "Actual_G3": actual
        })

    except Exception as e:
        logging.error(str(e))
        return jsonify({
            "Error":str(e)
        })

# Run Flask 

if __name__ == "__main__":
    app.run(debug=True)
    

 