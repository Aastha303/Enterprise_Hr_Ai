"""
16 - Employee Intelligence Table
Everything from Days 2-3 lands here: one row per employee pulling together
attrition risk, role, skill gaps, and the recommendation. This table IS the
business output of the project - the dashboard is just a view onto it.

Note: engagement (from hr_performance_engagement.csv / Cleaned_HR_Data_Analysis.csv)
is intentionally NOT joined in here - per docs/data_relationships.md it's a
different, unrelated employee population and cannot be merged on ID.
"""
import pandas as pd
import joblib

attr_features = pd.read_csv("../data/processed/attrition_features.csv")
attr_raw = pd.read_csv("../data/processed/employee_attrition_processed.csv")
gaps = pd.read_csv("../data/processed/employee_skill_gaps.csv")
recs = pd.read_csv("../data/processed/employee_recommendations.csv")

pipe = joblib.load("../models/v1/attrition_pipeline.joblib")
X = attr_features.drop(columns=["Attrition"])
probs = pipe.predict_proba(X)[:, 1]


def risk_bucket(p):
    if p >= 0.6:
        return "HIGH"
    if p >= 0.3:
        return "MEDIUM"
    return "LOW"


intelligence = pd.DataFrame({
    "EmployeeNumber": attr_raw["EmployeeNumber"],
    "Department": attr_raw["Department"],
    "JobRole": attr_raw["JobRole"],
    "Attrition_Prob": probs.round(3),
    "Risk": [risk_bucket(p) for p in probs],
})
intelligence = intelligence.merge(
    gaps[["EmployeeNumber", "occupation_title", "gap_count", "skill_gap"]], on="EmployeeNumber", how="left"
)
intelligence = intelligence.merge(recs, on="EmployeeNumber", how="left")

intelligence.to_csv("../data/processed/employee_intelligence.csv", index=False)

print("=== Employee Intelligence Table (sample) ===")
print(intelligence.head(10).to_string(index=False))
print(f"\n{len(intelligence)} rows -> data/processed/employee_intelligence.csv")
print("\nRisk distribution:")
print(intelligence["Risk"].value_counts())
