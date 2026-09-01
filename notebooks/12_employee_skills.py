"""
12 - Employee Skills Table
None of the 5 source datasets record what skills an employee CURRENTLY has -
essential_skills/software_skills describe role requirements, not a person's
actual competence. Without that, a real skill-gap calc is impossible.

This builds a controlled, explicitly-labelled placeholder: each employee is
assigned a plausible subset (60-90%) of their mapped occupation's top-importance
skills, seeded deterministically by employee ID. This is NOT real data - swap
in actual HR/LMS skill records in production.
"""
import pandas as pd
import numpy as np

attr = pd.read_csv("../data/processed/employee_attrition_processed.csv")
role_map = pd.read_csv("../data/processed/role_occupation_map.csv")
ess = pd.read_csv("../data/processed/essential_skills_processed.csv")
sw = pd.read_csv("../data/processed/software_skills_processed.csv")

attr = attr.merge(role_map, left_on="JobRole", right_on="job_role", how="left")

# Top skills per occupation: combine essential (core) skills and in-demand software/tools.
top_essential = (ess.sort_values("importance_score", ascending=False)
                  .groupby("occupation_code")["skill_name"].apply(lambda s: list(s.head(6))))
top_software = (sw[sw["In Demand"] == "Y"]
                 .groupby("occupation_code")["tool_name"].apply(lambda s: list(s.head(4))))

rows = []
rng_master = np.random.default_rng(42)
for _, emp in attr.iterrows():
    code = emp["occupation_code"]
    required = list(top_essential.get(code, [])) + list(top_software.get(code, []))
    if not required:
        continue
    # deterministic per-employee randomness so re-runs are reproducible
    rng = np.random.default_rng(int(emp["EmployeeNumber"]))
    keep_frac = rng.uniform(0.5, 0.9)
    n_keep = max(1, int(round(len(required) * keep_frac)))
    has = rng.choice(required, size=n_keep, replace=False)
    for skill in has:
        rows.append({"EmployeeNumber": emp["EmployeeNumber"], "current_skill": skill})

employee_skills = pd.DataFrame(rows)
employee_skills.to_csv("../data/processed/employee_skills_SYNTHETIC.csv", index=False)

print(f"Built synthetic current-skills table: {len(employee_skills)} (employee, skill) rows "
      f"across {employee_skills['EmployeeNumber'].nunique()} employees")
print("Saved -> data/processed/employee_skills_SYNTHETIC.csv")
print("\nFILENAME IS DELIBERATE: this is placeholder data, not observed employee skills.")
