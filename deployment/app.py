from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("models/iris_model.pkl")

# Define input schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Initialize app
app = FastAPI(title=" Iris Classifier API", version="1.0")

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
    return {"prediction": prediction}

