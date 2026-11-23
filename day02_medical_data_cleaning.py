import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 1. Clinical patient dataset with a missing glucose value (NaN) and an extreme age outlier (250)
df = pd.DataFrame({
    'Glucose': [85.0, 92.0, np.nan, 185.0, 200.0, 215.0],
    'Age':,
    'Diabetes': [0, 0, 0, 1, 1, 1]
})

# 2. Replace unrealistic age outlier (> 120) with the median of valid ages
median_age = df[df['Age'] < 120]['Age'].median()
df.loc[df['Age'] > 120, 'Age'] = median_age

# 3. Impute missing glucose value using the column median
median_glucose = df['Glucose'].median()
df['Glucose'] = df['Glucose'].fillna(median_glucose)

# 4. Train a Decision Tree classifier on the cleaned dataset
X = df[['Glucose', 'Age']]
y = df['Diabetes']

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# 5. Predict outcome for a new patient
prediction = model.predict([[195.0, 48]])
print("Cleaned Dataset:\n", df)
print("\nPrediction for new patient (Glucose=195, Age=48):", prediction[0])
