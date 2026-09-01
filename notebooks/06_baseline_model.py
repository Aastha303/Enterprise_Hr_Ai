"""
06 - Baseline Model
Logistic Regression on purpose: fast, explainable, gives real probabilities
(needed later to bucket employees into risk levels, not just yes/no).
Judged on precision/recall/F1/ROC-AUC - NOT accuracy, since attrition is imbalanced.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("../data/processed/attrition_features.csv")
y = df["Attrition"]
X = df.drop(columns=["Attrition"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train_scaled, y_train)

probs = model.predict_proba(X_test_scaled)[:, 1]
preds = model.predict(X_test_scaled)

print("=== Baseline: Logistic Regression ===")
print(classification_report(y_test, preds, target_names=["Stayed", "Left"]))
print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))
