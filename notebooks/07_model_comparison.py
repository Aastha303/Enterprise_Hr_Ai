"""
07 - Model Comparison
Same split, same preprocessing, three models. Pick a winner based on the
actual cost of mistakes: missing a genuinely high-risk employee is expensive,
so lean toward recall on the "Left" class over a marginally higher accuracy.
"""
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("../data/processed/attrition_features.csv")
y = df["Attrition"]
X = df.drop(columns=["Attrition"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)),
    ]),
    "XGBoost": Pipeline([
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric="logloss", random_state=42,
        )),
    ]),
}

rows = []
fitted = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)
    rows.append({
        "model": name,
        "precision_left": round(precision_score(y_test, preds), 3),
        "recall_left": round(recall_score(y_test, preds), 3),
        "f1_left": round(f1_score(y_test, preds), 3),
        "roc_auc": round(roc_auc_score(y_test, probs), 3),
    })

comparison = pd.DataFrame(rows).sort_values("recall_left", ascending=False)
print("=== Model Comparison (test set, 'Left' class) ===")
print(comparison.to_string(index=False))

# Pick winner on recall_left first (missed leavers are the expensive mistake), ROC-AUC as tiebreak.
winner_name = comparison.iloc[0]["model"]
print(f"\nWinner: {winner_name} (highest recall on the 'Left' class)")

os.makedirs("../models", exist_ok=True)
joblib.dump(fitted[winner_name], "../models/attrition_pipeline.joblib")
comparison.to_csv("../docs/model_comparison.csv", index=False)
print("Saved winning pipeline -> models/attrition_pipeline.joblib")
print("Saved comparison table -> docs/model_comparison.csv")

with open("../docs/model_winner.txt", "w") as f:
    f.write(winner_name)
