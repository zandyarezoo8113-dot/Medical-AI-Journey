import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Clinical training data: [Glucose (mg/dL), Age]
X = np.array([, , , , , ])
y = np.array()

# 2. Train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# 3. Evaluate a borderline patient
new_patient = np.array([])
default_pred = model.predict(new_patient)
probabilities = model.predict_proba(new_patient)

sick_prob = probabilities
print(f"Default Prediction (Threshold = 0.50): {default_pred}")
print(f"Predicted Probability [Healthy, Diabetic]: {np.round(probabilities * 100, 1)}%")

# 4. Apply a safer clinical threshold (0.35) to prevent False Negatives
clinical_threshold = 0.35
if sick_prob > clinical_threshold:
    print("⚠️ High Risk Alert: Further medical screening required.")
else:
    print("✅ Low Risk: Healthy.")
