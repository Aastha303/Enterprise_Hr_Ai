"""
Career Path Simulation (deck slide 12): given an employee's current role,
what's the next logical step, and how ready are they for it right now -
based on how much of the TARGET role's required skills they already have.

Uses the same synthetic current-skills table as the skill gap engine, so
the caveat is the same: this is a controlled placeholder, not observed
employee competence. The readiness % calculation itself is real logic.
"""
import json
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAREER_PATHS_FILE = os.path.join(BASE, "data", "external", "career_paths.json")
ROLE_MAP_FILE = os.path.join(BASE, "data", "processed", "role_occupation_map.csv")
ESSENTIAL_SKILLS_FILE = os.path.join(BASE, "data", "processed", "essential_skills_processed.csv")
SOFTWARE_SKILLS_FILE = os.path.join(BASE, "data", "processed", "software_skills_processed.csv")
EMPLOYEE_SKILLS_FILE = os.path.join(BASE, "data", "processed", "employee_skills_SYNTHETIC.csv")
ATTRITION_FILE = os.path.join(BASE, "data", "processed", "employee_attrition_processed.csv")
NAMES_FILE = os.path.join(BASE, "data", "processed", "employee_names_SYNTHETIC.csv")


def _required_skills_for(occupation_code: str, ess: pd.DataFrame, sw: pd.DataFrame) -> set:
    top_essential = set(ess[ess["occupation_code"] == occupation_code]
                         .sort_values("importance_score", ascending=False)["skill_name"].head(6))
    top_software = set(sw[(sw["occupation_code"] == occupation_code) & (sw["In Demand"] == "Y")]
                        ["tool_name"].head(4))
    return top_essential | top_software


def get_career_path(employee_id: int) -> dict:
    with open(CAREER_PATHS_FILE) as f:
        paths = json.load(f)

    attr = pd.read_csv(ATTRITION_FILE)
    row = attr[attr["EmployeeNumber"] == employee_id]
    if row.empty:
        return {"error": f"Employee {employee_id} not found"}
    current_role = row.iloc[0]["JobRole"]

    names = pd.read_csv(NAMES_FILE)
    name_row = names[names["EmployeeNumber"] == employee_id]
    employee_name = name_row.iloc[0]["EmployeeName"] if not name_row.empty else None

    next_role = paths.get(current_role)
    if next_role is None:
        return {
            "employee_id": employee_id,
            "employee_name": employee_name,
            "current_role": current_role,
            "next_role": None,
            "message": "No defined next step in the current career-path reference set for this role.",
        }

    role_map = pd.read_csv(ROLE_MAP_FILE)
    ess = pd.read_csv(ESSENTIAL_SKILLS_FILE)
    sw = pd.read_csv(SOFTWARE_SKILLS_FILE)
    has_skills = pd.read_csv(EMPLOYEE_SKILLS_FILE)

    # next_role is a job-title-style string, not an attrition JobRole, so map it
    # to an O*NET occupation the same way JobRole gets mapped (nearest match by name).
    match = role_map[role_map["job_role"].str.lower() == next_role.lower()]
    if match.empty:
        # fall back to substring match against occupation titles already resolved
        match = role_map[role_map["occupation_title"].str.contains(next_role.split()[0], case=False, na=False)]
    if match.empty:
        return {
            "employee_id": employee_id,
            "employee_name": employee_name,
            "current_role": current_role,
            "next_role": next_role,
            "message": f"'{next_role}' isn't mapped to an O*NET occupation yet - readiness can't be computed.",
        }

    target_code = match.iloc[0]["occupation_code"]
    target_title = match.iloc[0]["occupation_title"]
    required = _required_skills_for(target_code, ess, sw)

    has = set(has_skills[has_skills["EmployeeNumber"] == employee_id]["current_skill"])
    have_of_required = has & required
    readiness_pct = round(100 * len(have_of_required) / len(required), 1) if required else 0.0

    return {
        "employee_id": employee_id,
        "employee_name": employee_name,
        "current_role": current_role,
        "next_role": next_role,
        "next_role_occupation_title": target_title,
        "readiness_pct": readiness_pct,
        "skills_have": sorted(have_of_required),
        "skills_missing": sorted(required - has),
    }
