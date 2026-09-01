"""Serves the pre-computed skill gap tables (org-wide + per-employee)."""
import pandas as pd
from app.utils.config import ORG_SKILL_GAP_PATH, EMPLOYEE_INTELLIGENCE_PATH


def get_organization_skill_gaps() -> list:
    df = pd.read_csv(ORG_SKILL_GAP_PATH)
    return df.to_dict(orient="records")


def get_employee_skill_gap(employee_number: int) -> dict | None:
    df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)
    row = df[df["EmployeeNumber"] == employee_number]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "EmployeeNumber": int(r["EmployeeNumber"]),
        "EmployeeName": r["EmployeeName"],
        "JobRole": r["JobRole"],
        "occupation_title": r["occupation_title"],
        "gap_count": int(r["gap_count"]) if pd.notnull(r["gap_count"]) else None,
        "skill_gap": r["skill_gap"] if pd.notnull(r["skill_gap"]) else "",
    }
