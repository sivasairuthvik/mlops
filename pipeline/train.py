import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv("processed_student_data.csv")

X=df.drop(columns=["G3"])
y=df["G3"]

# Convert categorical columns to numerical values
X=pd.get_dummies(X)

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model=RandomForestRegressor(
    random_state=42
)

model.fit(X_train,y_train)

# Create model directory
os.makedirs("model",exist_ok=True)

# Save model
joblib.dump(model,"model/deployment.pkl")

# Save feature names for prediction
joblib.dump(X.columns.tolist(),"model/features.pkl")

print("Model Training Completed")
print("Model saved: model/deployment.pkl")