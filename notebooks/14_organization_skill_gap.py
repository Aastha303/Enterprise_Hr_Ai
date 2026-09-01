"""
14 - Organization-Wide Skill Gap
Same logic, rolled up across every employee: which skills are missing
organisation-wide, not just per person. Severity: 100+ missing -> HIGH,
50+ -> MEDIUM, else LOW. Feeds the dashboard's "critical skill gaps" chart.
"""
import pandas as pd

gap_df = pd.read_csv("../data/processed/employee_skill_gaps.csv")
gap_df["skill_gap"] = gap_df["skill_gap"].fillna("")

exploded = gap_df.assign(skill=gap_df["skill_gap"].str.split(", ")).explode("skill")
exploded = exploded[exploded["skill"] != ""]

org_gap = exploded.groupby("skill").size().sort_values(ascending=False).reset_index(name="employees_missing")


def severity(n):
    if n >= 100:
        return "HIGH"
    if n >= 50:
        return "MEDIUM"
    return "LOW"


org_gap["severity"] = org_gap["employees_missing"].apply(severity)

print("=== Organization-wide skill gaps ===")
print(org_gap.to_string(index=False))

org_gap.to_csv("../docs/organization_skill_gap.csv", index=False)
print("\nSaved -> docs/organization_skill_gap.csv")
