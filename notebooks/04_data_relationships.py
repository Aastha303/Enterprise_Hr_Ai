"""
04 - Data Relationships
Decide, honestly, how the processed tables actually connect.
A matching column name is not proof of a matching key - check overlap
before claiming a join is valid.
"""
import pandas as pd

P = "../data/processed"

attr = pd.read_csv(f"{P}/employee_attrition_processed.csv")
eng = pd.read_csv(f"{P}/engagement_processed.csv")
occ = pd.read_csv(f"{P}/occupation_master.csv")
ess = pd.read_csv(f"{P}/essential_skills_processed.csv")
sw = pd.read_csv(f"{P}/software_skills_processed.csv")

print("employee_attrition JobRole values:", attr["JobRole"].unique())
print("\nengagement JobRole values:", eng["JobRole"].unique())
print("\noccupation_master sample titles:", occ["occupation_title"].sample(5, random_state=1).tolist())

# essential_skills / software_skills share occupation_code with occupation_master - real key.
ess_join_ok = ess["occupation_code"].isin(occ["occupation_code"]).mean()
sw_join_ok = sw["occupation_code"].isin(occ["occupation_code"]).mean()
print(f"\nessential_skills -> occupation_master join coverage: {ess_join_ok:.1%}")
print(f"software_skills -> occupation_master join coverage: {sw_join_ok:.1%}")

# attrition vs engagement: same-shaped ID columns (EmployeeNumber / EmployeeID) but
# these are two DIFFERENT synthetic datasets from different sources - matching numbers
# would NOT refer to the same person. Confirm they're not the same population before
# ever joining them directly.
print(f"\nattrition rows: {len(attr)}, engagement rows: {len(eng)}")
print("attrition Departments:", sorted(attr['Department'].unique()))
print("engagement Departments:", sorted(eng['Department'].unique()))
