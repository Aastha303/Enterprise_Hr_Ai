"""
11 - Role Intelligence
occupation_master becomes the role reference table. Since neither HR dataset
uses O*NET occupation codes, map each JobRole string to its nearest O*NET
occupation title (fuzzy match) so downstream steps can say "ML Engineer
requires Python, MLOps, Docker" unambiguously. This is an approximation,
flagged with a similarity score - not treated as an exact key.
"""
import pandas as pd
from difflib import SequenceMatcher

occ = pd.read_csv("../data/processed/occupation_master.csv")
attr = pd.read_csv("../data/processed/employee_attrition_processed.csv")
eng = pd.read_csv("../data/processed/engagement_processed.csv")

# A small manual override table for common roles that fuzzy string match badly
# but have an obvious real-world O*NET equivalent (e.g. "Developer" vs "Software Developers").
MANUAL_OVERRIDES = {
    "developer": "Software Developers",
    "tester": "Software Quality Assurance Analysts and Testers",
    "engineer": "Software Developers",
    "support engineer": "Computer User Support Specialists",
    "helpdesk": "Computer User Support Specialists",
    "hr executive": "Human Resources Specialists",
    "hr manager": "Human Resources Managers",
    "seo analyst": "Market Research Analysts and Marketing Specialists",
    "content lead": "Public Relations Specialists",
    "account manager": "Sales Managers",
    "sales executive": "Sales Managers",
    "sales representative": "Sales Representatives of Services, Except Advertising, Insurance, Financial Services, and Travel",
    "research scientist": "Natural Sciences Managers",
    "research director": "Natural Sciences Managers",
    "laboratory technician": "Chemical Technicians",
    "manufacturing director": "Industrial Production Managers",
    "healthcare representative": "Sales Representatives, Wholesale and Manufacturing, Technical and Scientific Products",
    "human resources": "Human Resources Specialists",
    "manager": "General and Operations Managers",
    "auditor": "Accountants and Auditors",
    "accountant": "Accountants and Auditors",
}


def best_match(job_role: str):
    key = job_role.strip().lower()
    if key in MANUAL_OVERRIDES:
        target = MANUAL_OVERRIDES[key]
        row = occ[occ["occupation_title"] == target].iloc[0]
        return row["occupation_code"], row["occupation_title"], 1.0  # manual = high confidence

    scores = occ["occupation_title"].apply(lambda t: SequenceMatcher(None, key, t.lower()).ratio())
    best_idx = scores.idxmax()
    return occ.loc[best_idx, "occupation_code"], occ.loc[best_idx, "occupation_title"], round(scores.max(), 3)


all_roles = sorted(set(attr["JobRole"].unique()) | set(eng["JobRole"].unique()))
mapping_rows = []
for role in all_roles:
    code, title, score = best_match(role)
    mapping_rows.append({"job_role": role, "occupation_code": code, "occupation_title": title, "match_confidence": score})

role_map = pd.DataFrame(mapping_rows)
print("=== JobRole -> O*NET occupation mapping ===")
print(role_map.to_string(index=False))

role_map.to_csv("../data/processed/role_occupation_map.csv", index=False)
print("\nSaved -> data/processed/role_occupation_map.csv")
print("\nNote: match_confidence < 0.5 rows came from difflib string similarity, not a manual override -")
print("review those before trusting the skill-gap results for those roles.")
