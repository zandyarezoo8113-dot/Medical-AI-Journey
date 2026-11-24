import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. Patient dataset: [Glucose Level (large scale), Clinical Symptoms Count (small scale)]
X = np.array((85, 0, 99, 1, 110, 0, 189, 3, 200, 3, 210, 4)).reshape(6, 2)
Y = np.array((0, 0, 0, 1, 1, 1))

# 2. Apply Z-Score Standardization to prevent large features from dominating distance calculations
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, Y)

# 4. Transform and predict for a borderline patient (Glucose=140, Symptoms=4)
new_patient = np.array((140, 4)).reshape(1, 2)
new_patient_scaled = scaler.transform(new_patient)

diagnosis = knn.predict(new_patient_scaled)
print("KNN Diagnosis (0=Healthy, 1=High Risk):", diagnosis)
