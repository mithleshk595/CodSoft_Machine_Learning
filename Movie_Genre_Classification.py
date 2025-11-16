# ------------------------------
# Movie Genre Classification (CodSoft Internship Task 1)
# ------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
# ------------------------------
df = pd.read_csv("Movie.csv")   
# Check first few rows
print(df.head())

# 2. Select Input (X) & Output (y)
# ------------------------------
X = df['plot']          # plot / summary column
y = df['genre']         # genre column

# 3. Split Dataset
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Convert Text → TF-IDF Features
# ------------------------------
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5. Train Model (Logistic Regression)
# ------------------------------
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# 6. Predictions
# ------------------------------
y_pred = model.predict(X_test_tfidf)

# 7. Evaluation
# ------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Test with custom input
user_plot = "A superhero fights evil forces to save the world."
user_plot_tfidf = tfidf.transform([user_plot])
prediction = model.predict(user_plot_tfidf)

print("\nPredicted Genre:", prediction[0])
