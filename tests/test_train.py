import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pytest

def test_model_training():
    # Load model
    model = joblib.load('../models/iris_model.pkl')

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Predict
    y_pred = model.predict(X)

    # Basic accuracy check
    acc = accuracy_score(y, y_pred)
    assert acc > 0.9, f"Model accuracy too low: {acc:.2f}"

def test_model_output_shape():
    model = joblib.load('../models/iris_model.pkl')
    iris = load_iris()
    X = iris.data
    y_pred = model.predict(X)
    assert len(y_pred) == len(X), "Output shape mismatch"

