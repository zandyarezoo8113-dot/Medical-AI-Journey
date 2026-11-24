import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# 1. Load real-world Wisconsin Breast Cancer clinical dataset (569 patients, 30 features)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0 = Malignant (بدخیم), 1 = Benign (خوش‌خیم)

print(f"Dataset Loaded: {X.shape} patients and {X.shape} clinical features.")

# 2. Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Apply Feature Scaling (fit on train only to prevent Data Leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Logistic Regression model on real patient data
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. Evaluate model performance
predictions = model.predict(X_test_scaled)
acc = accuracy_score(y_test, predictions)
rec = recall_score(y_test, predictions)

print(f"Accuracy on Real Clinical Data: {acc * 100:.2f}%")
print(f"Recall (Sensitivity): {rec * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
