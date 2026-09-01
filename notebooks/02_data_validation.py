"""
02 - Data Validation
Define what "valid" means for each dataset before cleaning it.
Plain pandas assertions for the MVP; swap in Pandera later if rules
need to live in one place instead of scattered across scripts.
"""
import pandas as pd

DATA_PATH = "../data/raw"


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    return condition


results = []

# --- employee_attrition.csv ---
df = pd.read_csv(f"{DATA_PATH}/employee_attrition.csv", encoding="utf-8-sig")
results.append(check("schema: expected columns present",
                      {"Age", "Attrition", "Department", "JobRole", "MonthlyIncome"}.issubset(df.columns)))
results.append(check("type: Age is numeric", pd.api.types.is_integer_dtype(df["Age"])))
results.append(check("range: Age between 18-100", df["Age"].between(18, 100).all()))
results.append(check("uniqueness: EmployeeNumber never repeats", df["EmployeeNumber"].is_unique))
results.append(check("category: Attrition only Yes/No",
                      set(df["Attrition"].unique()) <= {"Yes", "No"}))
results.append(check("range: JobSatisfaction 1-4", df["JobSatisfaction"].between(1, 4).all()))

# --- hr_performance_engagement.csv (employee_performance_pro) ---
perf = pd.read_csv(f"{DATA_PATH}/hr_performance_engagement.csv", encoding="utf-8-sig")
results.append(check("schema: expected columns present",
                      {"EmployeeID", "Department", "JobRole", "PerformanceRating"}.issubset(perf.columns)))
results.append(check("uniqueness: EmployeeID never repeats", perf["EmployeeID"].is_unique))
results.append(check("range: WorkLifeBalanceScore 0-100 (observed scale)",
                      perf["WorkLifeBalanceScore"].between(0, 100).all(),
                      f"(min={perf['WorkLifeBalanceScore'].min()}, max={perf['WorkLifeBalanceScore'].max()})"))
results.append(check("range: Age between 18-100", perf["Age"].between(18, 100).all()))

# --- occupation_data.csv ---
occ = pd.read_csv(f"{DATA_PATH}/occupation_data.csv", encoding="utf-8-sig")
results.append(check("uniqueness: O*NET-SOC Code never repeats", occ["O*NET-SOC Code"].is_unique))
results.append(check("no nulls in Title", occ["Title"].notnull().all()))

# --- essential_skills.csv ---
skills = pd.read_csv(f"{DATA_PATH}/essential_skills.csv", encoding="utf-8-sig")
results.append(check("range: Data Value (Importance/Level) is plausible 0-10",
                      skills["Data Value"].between(0, 10).all(),
                      f"(min={skills['Data Value'].min()}, max={skills['Data Value'].max()})"))
results.append(check("every O*NET-SOC Code exists in occupation_data",
                      skills["O*NET-SOC Code"].isin(occ["O*NET-SOC Code"]).all()))

# --- software_skills.csv ---
sw = pd.read_csv(f"{DATA_PATH}/software_skills.csv", encoding="utf-8-sig")
results.append(check("every O*NET-SOC Code exists in occupation_data",
                      sw["O*NET-SOC Code"].isin(occ["O*NET-SOC Code"]).all()))
results.append(check("category: Hot Technology only Y/N",
                      set(sw["Hot Technology"].dropna().unique()) <= {"Y", "N"}))

print(f"\n{sum(results)}/{len(results)} checks passed")
if not all(results):
    print("WARNING: one or more validation checks failed - inspect before cleaning.")
