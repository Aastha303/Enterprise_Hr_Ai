"""Serves per-employee upskilling recommendations."""
import pandas as pd
from app.utils.config import EMPLOYEE_INTELLIGENCE_PATH, ATTRITION_PROCESSED_PATH, SKILL_GAP_PATH, ORG_SKILL_GAP_PATH


def get_all_recommendations() -> list:
    df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)
    return df[["EmployeeNumber", "EmployeeName", "JobRole", "recommendation"]].to_dict(orient="records")


def get_employee_recommendation(employee_number: int) -> dict | None:
    df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)
    row = df[df["EmployeeNumber"] == employee_number]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"EmployeeNumber": int(r["EmployeeNumber"]), "EmployeeName": r["EmployeeName"],
            "recommendation": r["recommendation"]}


def list_employees() -> list:
    """Lightweight roster for populating name/ID pickers and dashboard charts in the frontend."""
    df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)
    income = pd.read_csv(ATTRITION_PROCESSED_PATH)[["EmployeeNumber", "MonthlyIncome"]]
    df = df.merge(income, on="EmployeeNumber", how="left")
    cols = ["EmployeeNumber", "EmployeeName", "Department", "JobRole", "Risk", "Attrition_Prob", "MonthlyIncome"]
    return df[cols].sort_values("EmployeeNumber").to_dict(orient="records")


def get_raw_employee_record(employee_number: int) -> dict | None:
    """Full raw attrition-model input record for one employee - used by the
    What-If Policy Simulator to build a baseline it can then perturb."""
    df = pd.read_csv(ATTRITION_PROCESSED_PATH)
    row = df[df["EmployeeNumber"] == employee_number]
    if row.empty:
        return None
    names = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)[["EmployeeNumber", "EmployeeName"]]
    row = row.merge(names, on="EmployeeNumber", how="left")
    record = row.iloc[0].where(pd.notnull(row.iloc[0]), None).to_dict()
    # Attrition column is the training label (Yes/No -> not part of a prediction request), drop it
    record.pop("Attrition", None)
    return record


def get_full_employee_record(employee_number: int) -> dict | None:
    """The full intelligence record for one person - attrition risk, role, skill gap, recommendation."""
    df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)
    row = df[df["EmployeeNumber"] == employee_number]
    if row.empty:
        return None
    return row.iloc[0].where(pd.notnull(row.iloc[0]), None).to_dict()


def get_recommendation_quality_metrics() -> dict:
    """
    Get recommendation quality metrics (Precision@k, Recall@k, MRR@k).
    
    Returns:
        Dictionary with:
        - summary: Aggregate metrics across all employees
        - evaluation_details: Path to detailed CSV report
    """
    try:
        from app.services.recommendation_evaluation_service import get_recommendation_quality_report
        
        gap_df = pd.read_csv(SKILL_GAP_PATH)
        org_gap_df = pd.read_csv(ORG_SKILL_GAP_PATH)
        
        report = get_recommendation_quality_report(gap_df, org_gap_df)
        
        return {
            "status": "success",
            "summary_statistics": report["summary_statistics"],
            "total_employees_evaluated": report["total_employees_evaluated"],
            "mean_skills_per_employee": report["mean_skills_per_employee"],
            "evaluation_details_file": "docs/recommendation_evaluation.csv"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "note": "Evaluation metrics require skill gap data. Run notebook 20_recommendation_evaluation.py first."
        }
