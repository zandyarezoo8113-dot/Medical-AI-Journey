import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.array([[85, 22], [92, 26], [110, 30], [185, 45], [200, 52], [215, 60]])
y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(x, y)

new_patient = np.array([[145, 38]])

predict1 = model.predict(new_patient)
predict2 = model.predict_proba(new_patient)

print("تشخیص قطعی (0=سالم، 1=بیمار):", predict1)
print("درصد احتمال [سالم, بیمار]:", np.round(predict2 * 100, 1))
sick_prob=predict2[0][1]
if sick_prob > 0.35:
  print("نیاز به آزمایش تکمیلی(پرخطر)⚠")
else:
  print("سالم✅")
