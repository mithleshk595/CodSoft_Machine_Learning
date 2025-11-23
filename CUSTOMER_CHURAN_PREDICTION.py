# -----------------------------------------------------------
# CUSTOMER CHURN PREDICTION (RANDOM FOREST + VISUALIZATION)
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# -----------------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------------
data = pd.read_csv("customer_churn.csv")

print("\nDataset Loaded Successfully")
print(data.head())

# -----------------------------------------------------------
# 2. Encode Categorical Columns
# -----------------------------------------------------------
le = LabelEncoder()
for col in data.select_dtypes(include='object'):
    data[col] = le.fit_transform(data[col])

# -----------------------------------------------------------
# 3. Visualization
# -----------------------------------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x=data["Churn"])
plt.title("Churn Count")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=data["MonthlyCharges"], y=data["TotalCharges"], hue=data["Churn"])
plt.title("Monthly Charges vs Total Charges")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=data["Churn"], y=data["Age"])
plt.title("Age vs Churn")
plt.show()

# -----------------------------------------------------------
# 4. Split Data
# -----------------------------------------------------------

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 5. Train Random Forest Model
# -----------------------------------------------------------
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------------------------------
# 6. Predict
# -----------------------------------------------------------
y_pred = model.predict(X_test)

# -----------------------------------------------------------
# 7. Evaluate Model
# -----------------------------------------------------------
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------------------------------------
# 8. Feature Importance
# -----------------------------------------------------------
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp.nlargest(10).plot(kind='barh')
plt.title("Feature Importance - Top 10")
plt.show()

# -----------------------------------------------------------
# 9. Predict on Single Customer
# -----------------------------------------------------------
print("\nExample Input Row:")
print(X_test.iloc[0])

prediction = model.predict([X_test.iloc[0]])
print("\nPrediction (0 = No Churn, 1 = Yes Churn):", prediction[0])
