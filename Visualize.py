import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../data/customer_churn.csv")

sns.countplot(x=data["Churn"])
plt.title("Churn Count")
plt.show()

sns.scatterplot(x=data["MonthlyCharges"], y=data["TotalCharges"], hue=data["Churn"])
plt.title("Monthly Charges vs Total Charges")
plt.show()

sns.boxplot(x=data["Churn"], y=data["Age"])
plt.title("Age vs Churn")
plt.show()
