"""
Concrete tool implementations, registered against app.agents.tools.

Mirrors the deck's example tool-call sequence (slide 15):
get_employee_profile -> get_skills -> get_role_requirements ->
calculate_skill_gap -> recommend_courses -> generate_learning_plan

Plus the deck's exact governance example: get_all_employee_salary() is
gated to hr_admin, so an "employee" caller gets a PermissionDenied, not data.
"""
import pandas as pd
from app.agents.tools import register_tool
from app.utils.config import EMPLOYEE_INTELLIGENCE_PATH
from app.services.career_service import get_career_path
from app.rag.generator import answer_policy_question

INTEL_PATH = EMPLOYEE_INTELLIGENCE_PATH
ATTR_PATH = "data/processed/employee_attrition_processed.csv"


@register_tool("get_employee_profile", "Basic role/department info for one employee.", required_role="employee")
def get_employee_profile(employee_id: int) -> dict:
    df = pd.read_csv(INTEL_PATH)
    row = df[df["EmployeeNumber"] == employee_id]
    if row.empty:
        return {"error": f"Employee {employee_id} not found"}
    r = row.iloc[0]
    return {"EmployeeNumber": int(r["EmployeeNumber"]), "EmployeeName": r["EmployeeName"],
            "Department": r["Department"], "JobRole": r["JobRole"]}


@register_tool("get_attrition_risk", "Attrition probability and risk bucket for one employee.", required_role="manager")
def get_attrition_risk(employee_id: int) -> dict:
    df = pd.read_csv(INTEL_PATH)
    row = df[df["EmployeeNumber"] == employee_id]
    if row.empty:
        return {"error": f"Employee {employee_id} not found"}
    r = row.iloc[0]
    return {"EmployeeNumber": int(r["EmployeeNumber"]), "EmployeeName": r["EmployeeName"],
            "attrition_probability": float(r["Attrition_Prob"]), "risk": r["Risk"]}


@register_tool("calculate_skill_gap", "Missing skills for one employee's mapped occupation.", required_role="employee")
def calculate_skill_gap(employee_id: int) -> dict:
    df = pd.read_csv(INTEL_PATH)
    row = df[df["EmployeeNumber"] == employee_id]
    if row.empty:
        return {"error": f"Employee {employee_id} not found"}
    r = row.iloc[0]
    return {"EmployeeNumber": int(r["EmployeeNumber"]), "EmployeeName": r["EmployeeName"],
            "gap_count": int(r["gap_count"]) if pd.notnull(r["gap_count"]) else None,
            "skill_gap": r["skill_gap"] if pd.notnull(r["skill_gap"]) else ""}


@register_tool("recommend_courses", "Upskilling recommendation for one employee.", required_role="employee")
def recommend_courses(employee_id: int) -> dict:
    df = pd.read_csv(INTEL_PATH)
    row = df[df["EmployeeNumber"] == employee_id]
    if row.empty:
        return {"error": f"Employee {employee_id} not found"}
    r = row.iloc[0]
    return {"EmployeeNumber": int(r["EmployeeNumber"]), "EmployeeName": r["EmployeeName"],
            "recommendation": r["recommendation"]}


@register_tool("generate_learning_plan", "Career-path readiness + next-step plan for one employee.", required_role="employee")
def generate_learning_plan(employee_id: int) -> dict:
    return get_career_path(employee_id)


@register_tool("ask_policy", "Answer an HR policy question via RAG over policy documents.", required_role="employee")
def ask_policy(question: str) -> dict:
    return answer_policy_question(question)


@register_tool("get_all_employee_salary", "ALL employees' MonthlyIncome - sensitive, HR-admin only.", required_role="hr_admin")
def get_all_employee_salary() -> dict:
    df = pd.read_csv(ATTR_PATH)
    names = pd.read_csv(INTEL_PATH)[["EmployeeNumber", "EmployeeName"]]
    df = df.merge(names, on="EmployeeNumber", how="left")
    return df[["EmployeeNumber", "EmployeeName", "MonthlyIncome"]].to_dict(orient="records")
