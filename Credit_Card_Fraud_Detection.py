# credit_card_fraud_detection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
DATA_PATH = "creditcard.csv"  
RANDOM_STATE = 42

# --- LOAD DATA ---
df = pd.read_csv(DATA_PATH)
print("Data shape:", df.shape)
print(df['Class'].value_counts(normalize=True))

# --- QUICK EXPLORATION ---
print(df.describe().T[['mean','std']].head())

# Plot class imbalance
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title('Class distribution (0 = legit, 1 = fraud)')
plt.show()

# --- FEATURES / TARGET ---
X = df.drop(columns=['Class'])
y = df['Class']

# Convert 'Time' and 'Amount' with scaling (V1..V28 are already PCA components)
scaler = StandardScaler()
X[['Time','Amount']] = scaler.fit_transform(X[['Time','Amount']])

# --- Train-test split (stratified) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Train class distribution:", y_train.value_counts(normalize=True).to_dict())
print("Test class distribution:", y_test.value_counts(normalize=True).to_dict())

# --- Handle imbalance with SMOTE inside a pipeline for model training ---
smote = SMOTE(random_state=RANDOM_STATE, n_jobs=-1)

# We'll train three models: LogisticRegression, RandomForest, XGBoost
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=4)
}

results = {}

for name, model in models.items():
    print("\n--- Training", name, "---")
    pipeline = ImbPipeline(steps=[('smote', smote), ('clf', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1]

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    results[name] = {
        'model': pipeline,
        'confusion_matrix': cm,
        'report': report,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'y_proba': y_proba
    }

    print("Confusion matrix:\n", cm)
    print("ROC AUC:", round(roc_auc,4))
    print("PR AUC (avg precision):", round(pr_auc,4))
    print("Classification report:\n", report)

# --- Compare ROC curves ---
plt.figure(figsize=(8,6))
for name, info in results.items():
    fpr, tpr, _ = roc_curve(y_test, info['y_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={info['roc_auc']:.4f})")
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

# --- Show best model's confusion matrix heatmap (example: XGBoost) ---
best_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
best = results[best_name]
print("\nBest model by ROC AUC:", best_name)

plt.figure(figsize=(5,4))
sns.heatmap(best['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'{best_name} Confusion Matrix')
plt.show()


# --- Save the best model (optional) ---

import joblib
joblib.dump(best['model'], 'best_fraud_model.joblib')
print("Saved best model to best_fraud_model.joblib")
