"""
08 - Model Explainability (SHAP)
A prediction like "Employee 101 - 82% attrition risk" is useless without "why".
Global view: what generally drives attrition. Local view: why THIS employee is flagged.
"""
import pandas as pd
import joblib
import shap
import json

pipe = joblib.load("../models/attrition_pipeline.joblib")
df = pd.read_csv("../data/processed/attrition_features.csv")
y = df["Attrition"]
X = df.drop(columns=["Attrition"])

clf = pipe.named_steps["clf"]
# If the winning model has a scaler step, SHAP needs the transformed data to match what the model saw.
X_for_shap = X.copy()
if "scaler" in pipe.named_steps:
    X_for_shap = pd.DataFrame(pipe.named_steps["scaler"].transform(X), columns=X.columns)

sample = X_for_shap.sample(min(200, len(X_for_shap)), random_state=42)

try:
    explainer = shap.TreeExplainer(clf)
except Exception:
    explainer = shap.LinearExplainer(clf, sample) if hasattr(clf, "coef_") else shap.Explainer(clf, sample)

shap_values = explainer.shap_values(sample)
if isinstance(shap_values, list):  # some explainers return [class0, class1]
    shap_values = shap_values[1]

# --- Global importance ---
mean_abs_shap = pd.Series(abs(shap_values).mean(axis=0), index=sample.columns).sort_values(ascending=False)
print("=== Global drivers of attrition (top 10) ===")
print(mean_abs_shap.head(10))
mean_abs_shap.head(20).to_csv("../docs/shap_global_importance.csv", header=["mean_abs_shap"])

# --- Local example: explain one specific employee ---
example_idx = 0
row = sample.iloc[example_idx]
row_shap = pd.Series(shap_values[example_idx], index=sample.columns)
top3 = row_shap.abs().sort_values(ascending=False).head(3)

print("\n=== Local explanation: example employee ===")
local_explanation = {}
for feat in top3.index:
    direction = "increases" if row_shap[feat] > 0 else "decreases"
    print(f"  {feat}: {direction} risk (shap={row_shap[feat]:.3f})")
    local_explanation[feat] = {"shap_value": float(row_shap[feat]), "direction": direction}

with open("../docs/shap_local_example.json", "w") as f:
    json.dump(local_explanation, f, indent=2)

print("\nSaved -> docs/shap_global_importance.csv, docs/shap_local_example.json")
