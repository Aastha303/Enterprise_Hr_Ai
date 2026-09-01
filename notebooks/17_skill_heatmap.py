"""
17 - Organizational Skill Heatmap
The deck's leadership-intelligence view (slide 13): for each skill, how many
employees' roles REQUIRE it, how many employees HAVE it, and the resulting
gap. Different from 14_organization_skill_gap.py (which only counts missing
skills) - this one gives the full required/available/gap picture per skill,
which is what a heatmap or leadership chart actually needs.
"""
import pandas as pd

attr = pd.read_csv("../data/processed/employee_attrition_processed.csv")
role_map = pd.read_csv("../data/processed/role_occupation_map.csv")
ess = pd.read_csv("../data/processed/essential_skills_processed.csv")
sw = pd.read_csv("../data/processed/software_skills_processed.csv")
has_skills = pd.read_csv("../data/processed/employee_skills_SYNTHETIC.csv")

attr = attr.merge(role_map, left_on="JobRole", right_on="job_role", how="left")

top_essential = (ess.sort_values("importance_score", ascending=False)
                  .groupby("occupation_code")["skill_name"].apply(lambda s: list(s.head(6))))
top_software = (sw[sw["In Demand"] == "Y"]
                 .groupby("occupation_code")["tool_name"].apply(lambda s: list(s.head(4))))

has_by_emp = has_skills.groupby("EmployeeNumber")["current_skill"].apply(set).to_dict()

required_counter = {}
available_counter = {}

for _, emp in attr.iterrows():
    code = emp["occupation_code"]
    required = set(top_essential.get(code, [])) | set(top_software.get(code, []))
    has = has_by_emp.get(emp["EmployeeNumber"], set())
    for skill in required:
        required_counter[skill] = required_counter.get(skill, 0) + 1
        if skill in has:
            available_counter[skill] = available_counter.get(skill, 0) + 1

rows = []
for skill, required_n in required_counter.items():
    available_n = available_counter.get(skill, 0)
    rows.append({
        "skill": skill,
        "required": required_n,
        "available": available_n,
        "gap": required_n - available_n,
    })

heatmap = pd.DataFrame(rows).sort_values("gap", ascending=False)
print("=== Organizational Skill Heatmap (Required vs Available vs Gap) ===")
print(heatmap.to_string(index=False))

heatmap.to_csv("../docs/skill_heatmap.csv", index=False)
print("\nSaved -> docs/skill_heatmap.csv")
