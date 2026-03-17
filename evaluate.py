import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test data and model
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
model = joblib.load('models/disease_model.pkl')

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")