"""
01 - Data Understanding
Load every raw file and profile it: shape, columns, dtypes, missingness,
duplicates, candidate join keys. No modelling, no cleaning yet.
"""
import pandas as pd
import os
import json

pd.set_option("display.max_columns", None)
DATA_PATH = "../data/raw"

FILES = {
    "employee_attrition": "employee_attrition.csv",
    "hr_performance_engagement": "hr_performance_engagement.csv",
    "occupation_data": "occupation_data.csv",
    "essential_skills": "essential_skills.csv",
    "software_skills": "software_skills.csv",
}

print(os.listdir(DATA_PATH))  # sanity check - should list all 5 files

profile_report = {}

for name, fname in FILES.items():
    path = os.path.join(DATA_PATH, fname)
    df = pd.read_csv(path, encoding="utf-8-sig")

    id_like_cols = [c for c in df.columns if "id" in c.lower() or "code" in c.lower()]
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(2)

    print(f"\n=== {name} ({fname}) ===")
    print("shape:", df.shape)
    print("dtypes:\n", df.dtypes.value_counts())
    print("candidate id/code columns:", id_like_cols)
    print("duplicated rows:", df.duplicated().sum())
    print("top missing columns:\n", missing_pct.head(10))

    entry = {
        "file": fname,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "candidate_id_columns": id_like_cols,
        "duplicated_rows": int(df.duplicated().sum()),
        "missing_pct_top10": missing_pct.head(10).to_dict(),
    }

    if name == "employee_attrition" and "Attrition" in df.columns:
        balance = (df["Attrition"].value_counts(normalize=True) * 100).round(2).to_dict()
        print("Attrition target balance (%):", balance)
        entry["target_balance_pct"] = balance

    profile_report[name] = entry

os.makedirs("../docs", exist_ok=True)
with open("../docs/data_profile.json", "w") as f:
    json.dump(profile_report, f, indent=2)

print("\nSaved profile report -> docs/data_profile.json")
