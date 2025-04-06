from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Load the trained model
model = joblib.load("models/iris_model.pkl")

# Define input schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Initialize app
app = FastAPI(title="Iris Classifier API", version="1.0")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}

# Prediction endpoint
@app.post("/predict")
def predict_species(features: IrisFeatures):
    data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    prediction = model.predict(data)[0]

    # Log to monitoring file
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sepal_length": features.sepal_length,
        "sepal_width": features.sepal_width,
        "petal_length": features.petal_length,
        "petal_width": features.petal_width,
        "prediction": prediction
    }

    log_path = "monitoring/monitoring_logs.csv"
    os.makedirs("monitoring", exist_ok=True)
    if os.path.exists(log_path):
        pd.DataFrame([log_entry]).to_csv(log_path, mode='a', header=False, index=False)
    else:
        pd.DataFrame([log_entry]).to_csv(log_path, index=False)

    return {"prediction": prediction}

