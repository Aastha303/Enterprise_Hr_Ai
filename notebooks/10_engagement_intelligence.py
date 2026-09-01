"""
10 - Engagement Analytics
No ML here - just aggregation. Average engagement, by department, and the
lowest-engagement employees so HR can look at them directly.
"""
import pandas as pd

eng = pd.read_csv("../data/processed/engagement_processed.csv")

by_dept = eng.groupby("Department")["WorkLifeBalanceIndex"].mean().sort_values()
print("=== Avg Work-Life Balance Index by department ===")
print(by_dept.round(1))

lowest = eng.nsmallest(10, "WorkLifeBalanceIndex")[["EmployeeID", "Name", "Department", "JobRole", "WorkLifeBalanceIndex", "PerformanceRating"]]
print("\n=== 10 lowest-engagement employees ===")
print(lowest.to_string(index=False))

by_dept.to_csv("../docs/engagement_by_department.csv", header=["avg_worklife_balance_index"])
lowest.to_csv("../docs/lowest_engagement_employees.csv", index=False)
print("\nSaved -> docs/engagement_by_department.csv, docs/lowest_engagement_employees.csv")
