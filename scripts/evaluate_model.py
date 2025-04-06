import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the saved model
model = joblib.load('models/iris_model.pkl')

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Predict using the loaded model
y_pred = model.predict(X)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred, target_names=iris.target_names)
conf_matrix = confusion_matrix(y, y_pred)

print(f"\n Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n")
print(report)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved to confusion_matrix.png")

