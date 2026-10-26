import joblib
import pandas as pd

# Load dataset
df=pd.read_csv("processed_student_data.csv")

X=df.drop(columns=["G3"])

# Apply the same encoding used during training
X=pd.get_dummies(X)

# Load feature names
features=joblib.load("model/features.pkl")

# Match training features
X=X.reindex(columns=features,fill_value=0)

# Load model
model=joblib.load("model/deployment.pkl")

# Predict first student
prediction=model.predict(X.iloc[[0]])

print("Prediction:",round(prediction[0],2))
print("Actual:",df.iloc[0]["G3"])