"""Engagement analytics - descriptive only, no ML. Powers dashboard summary + charts."""
import pandas as pd
from app.utils.config import EMPLOYEE_INTELLIGENCE_PATH, ENGAGEMENT_PATH


def get_dashboard_summary() -> dict:
    intel = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)
    eng = pd.read_csv(ENGAGEMENT_PATH)
    return {
        "total_employees": int(len(intel)),
        "high_risk_employees": int((intel["Risk"] == "HIGH").sum()),
        "average_engagement_index": round(float(eng["WorkLifeBalanceIndex"].mean()), 1),
    }


def get_attrition_by_department() -> list:
    intel = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)
    grouped = (
        intel.groupby("Department")
        .agg(total=("EmployeeNumber", "count"),
             high_risk=("Risk", lambda s: (s == "HIGH").sum()),
             avg_attrition_prob=("Attrition_Prob", "mean"))
        .reset_index()
    )
    grouped["avg_attrition_prob"] = grouped["avg_attrition_prob"].round(3)
    return grouped.to_dict(orient="records")
