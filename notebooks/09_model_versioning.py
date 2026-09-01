"""
09 - Model Versioning
Simple versioning before reaching for MLflow: every trained model gets its
own folder + metadata.json so any prediction can be traced back to exactly
which model made it. Add MLflow later, once manual versioning starts to hurt.
"""
import json
import os
import shutil
import joblib
from datetime import date
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd

VERSION = "v1"
MODEL_DIR = f"../models/{VERSION}"
os.makedirs(MODEL_DIR, exist_ok=True)

winner_name = open("../docs/model_winner.txt").read().strip()
shutil.copy("../models/attrition_pipeline.joblib", f"{MODEL_DIR}/attrition_pipeline.joblib")

comparison = pd.read_csv("../docs/model_comparison.csv")
winner_row = comparison[comparison["model"] == winner_name].iloc[0]

metadata = {
    "model_name": "Attrition Prediction Model",
    "version": VERSION,
    "algorithm": winner_name,
    "training_date": str(date.today()),
    "roc_auc": float(winner_row["roc_auc"]),
    "f1_score": float(winner_row["f1_left"]),
    "recall_left": float(winner_row["recall_left"]),
    "precision_left": float(winner_row["precision_left"]),
    "training_rows": 1470,
    "target_column": "Attrition",
    "notes": "Class-imbalance handled via class_weight/scale_pos_weight; selected on recall for the minority 'Left' class.",
}

with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Versioned model saved -> models/{VERSION}/")
print(json.dumps(metadata, indent=2))
