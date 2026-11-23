import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

# 1. Clinical dataset: [Glucose (mg/dL), Age]
X = np.array([,
 ,
 ,
 ,
 ,
    [201, 40]
])
y = np.array([0, 0, 0, 1, 1, 1])

# 2. Split dataset into training (67%) and testing (33%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 3. Train Decision Tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Predict and evaluate on unseen test data
predictions = model.predict(X_test)

print("Model Predictions:", predictions)
print("Actual Labels (Ground Truth):", y_test)
print(f"Accuracy Score: {accuracy_score(y_test, predictions) * 100:.1f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
