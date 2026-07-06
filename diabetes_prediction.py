import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("diabetes.csv")


X = df[[
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]]
y = df["Outcome"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)


scaler = StandardScaler()


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# New patient
Pregnancies = int(input("Enter the Pregnancies :: "))
Glucose = int(input("Enter the Glucose :: "))
BloodPressure = int(input("Enter the BloodPressure :: "))
SkinThickness = int(input("Enter the SkinThickness :: "))
Insulin = int(input("Enter the Insulin :: "))
BMI = float(input("Enter the BMI :: "))
DiabetesPedigreeFunction = float(input("Enter the DiabetesPedigreeFunction :: "))
Age = int(input("Enter the Age :: "))



p = pd.DataFrame({
    "Pregnancies": [Pregnancies],
    "Glucose": [Glucose],
    "BloodPressure": [BloodPressure],
    "SkinThickness": [SkinThickness],
    "Insulin": [Insulin],
    "BMI": [BMI],
    "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
    "Age": [Age]
})

p_scaled = scaler.transform(p)



result = model.predict(p_scaled)[0]

if result == 1:
    print("Patient is Diabetic")
else:
    print("Patient is Not Diabetic")