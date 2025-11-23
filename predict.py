import pandas as pd
import pickle

# Load model
with open("../model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset to get columns
data = pd.read_csv("../data/customer_churn.csv")
X = data.drop("Churn", axis=1)

# Sample prediction
sample = X.iloc[0]
prediction = model.predict([sample])

print("Sample Input:", sample)
print("\nPrediction (0=No Churn, 1=Yes Churn):", prediction[0])
