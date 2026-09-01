"""
03 - Data Cleaning
Fix what validation flagged: standardize categories, dedupe skill-name
spelling variants, normalize dtypes. Save clean copies separately from
raw so raw data is never overwritten.
"""
import pandas as pd
import numpy as np
import os

RAW = "../data/raw"
OUT = "../data/processed"
os.makedirs(OUT, exist_ok=True)

# --- employee_attrition ---
df = pd.read_csv(f"{RAW}/employee_attrition.csv", encoding="utf-8-sig")
# EmployeeCount and StandardHours are constant across all rows (verified below) - dead weight, drop them.
constant_cols = [c for c in df.columns if df[c].nunique() == 1]
print("Dropping constant columns (no signal):", constant_cols)
df = df.drop(columns=constant_cols)
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
df.to_csv(f"{OUT}/employee_attrition_processed.csv", index=False)

# --- hr_performance_engagement ---
perf = pd.read_csv(f"{RAW}/hr_performance_engagement.csv", encoding="utf-8-sig")
# WorkLifeBalanceScore is actually a standardized (z-score-like) metric, not 0-100.
# Rescale to a 0-100 "WorkLifeBalanceIndex" so it's comparable to other 0-100 scores downstream.
wlb = perf["WorkLifeBalanceScore"]
perf["WorkLifeBalanceIndex"] = ((wlb - wlb.min()) / (wlb.max() - wlb.min()) * 100).round(1)
# CustomerSatisfaction is 64% missing (likely N/A for non-customer-facing roles) - keep but flag, don't impute blindly.
perf["CustomerSatisfaction_missing"] = perf["CustomerSatisfaction"].isnull()
perf.to_csv(f"{OUT}/engagement_processed.csv", index=False)

# --- occupation_data -> role master ---
occ = pd.read_csv(f"{RAW}/occupation_data.csv", encoding="utf-8-sig")
occ = occ.rename(columns={"O*NET-SOC Code": "occupation_code", "Title": "occupation_title"})
occ.to_csv(f"{OUT}/occupation_master.csv", index=False)

# --- essential_skills: keep Importance scale only, this is what matters for "required skills" ---
skills = pd.read_csv(f"{RAW}/essential_skills.csv", encoding="utf-8-sig")
skills = skills[skills["Scale ID"] == "IM"].copy()  # IM = Importance
skills = skills.rename(columns={
    "O*NET-SOC Code": "occupation_code",
    "Element Name": "skill_name",
    "Data Value": "importance_score",
})[["occupation_code", "skill_name", "importance_score"]]
skills.to_csv(f"{OUT}/essential_skills_processed.csv", index=False)

# --- software_skills: normalize messy tool-name spelling variants ---
sw = pd.read_csv(f"{RAW}/software_skills.csv", encoding="utf-8-sig")
sw = sw.rename(columns={
    "O*NET-SOC Code": "occupation_code",
    "Workplace Example": "tool_name",
    "Element Name": "tool_category",
})

# Normalize obvious duplicate spellings of the same tool (case, punctuation, common aliases).
ALIASES = {
    "amazon web services": "AWS",
    "aws cloud": "AWS",
    "aws": "AWS",
    "microsoft azure": "Azure",
    "google cloud platform": "Google Cloud Platform",
    "gcp": "Google Cloud Platform",
    "microsoft excel": "Microsoft Excel",
    "ms excel": "Microsoft Excel",
}


def normalize_tool_name(name: str) -> str:
    key = str(name).strip().lower()
    return ALIASES.get(key, str(name).strip())


sw["tool_name_clean"] = sw["tool_name"].apply(normalize_tool_name)
sw = sw[["occupation_code", "tool_name_clean", "tool_category", "Hot Technology", "In Demand"]]
sw = sw.rename(columns={"tool_name_clean": "tool_name"})
sw.to_csv(f"{OUT}/software_skills_processed.csv", index=False)

print("Saved 5 processed files to data/processed/:")
for f in os.listdir(OUT):
    print(" -", f)
