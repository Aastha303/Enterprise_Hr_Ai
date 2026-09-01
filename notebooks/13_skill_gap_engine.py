"""
13 - Skill Gap Engine
Core logic is set subtraction: required skills for the employee's mapped
occupation, minus what the employee already has. Weighted by importance
so a missing "nice to have" doesn't rank the same as a missing core skill.
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

gap_rows = []
for _, emp in attr.iterrows():
    code = emp["occupation_code"]
    required = set(top_essential.get(code, [])) | set(top_software.get(code, []))
    has = has_by_emp.get(emp["EmployeeNumber"], set())
    gap = required - has
    gap_rows.append({
        "EmployeeNumber": emp["EmployeeNumber"],
        "JobRole": emp["JobRole"],
        "occupation_title": emp["occupation_title"],
        "required_count": len(required),
        "has_count": len(has & required),
        "gap_count": len(gap),
        "skill_gap": ", ".join(sorted(gap)) if gap else "",
    })

gap_df = pd.DataFrame(gap_rows)
gap_df.to_csv("../data/processed/employee_skill_gaps.csv", index=False)

print("=== Example: one employee's skill gap ===")
example = gap_df.iloc[0]
print(f"EmployeeNumber {example['EmployeeNumber']} ({example['JobRole']} -> {example['occupation_title']})")
print(f"Required: {example['required_count']}, Has: {example['has_count']}, Gap: {example['skill_gap']}")

print(f"\nComputed skill gaps for {len(gap_df)} employees -> data/processed/employee_skill_gaps.csv")
print(f"Average gap size: {gap_df['gap_count'].mean():.1f} missing skills per employee")
