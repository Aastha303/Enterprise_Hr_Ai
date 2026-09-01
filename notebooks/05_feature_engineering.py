"""
05 - Feature Engineering
Turn employee_attrition_processed.csv into model-ready features.
Every engineered feature has a stated reason - no "seemed interesting" additions.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("../data/processed/employee_attrition_processed.csv")

# --- Engineered features (each with a reason) ---
# Income per year at company: normalizes pay against tenure - flags people paid low for how long they've stayed.
df["IncomePerYearAtCompany"] = df["MonthlyIncome"] / df["YearsAtCompany"].replace(0, 1)

# Gap since last promotion, relative to tenure: a raw "years since promotion" of 3 means
# very different things for someone who joined last year vs 10 years ago.
df["PromotionGapRatio"] = df["YearsSinceLastPromotion"] / df["YearsAtCompany"].replace(0, 1)

# Overall satisfaction score: JobSatisfaction, EnvironmentSatisfaction and RelationshipSatisfaction
# are all 1-4 scales measuring related-but-distinct things; averaging gives one composite signal
# instead of three correlated ones the model would have to re-learn are related.
df["OverallSatisfaction"] = df[["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"]].mean(axis=1)

# Experience ratio: how much of the person's total career has been at THIS company -
# a low ratio with high total experience can signal a recent, possibly still-settling, hire.
df["ExperienceRatio"] = df["YearsAtCompany"] / df["TotalWorkingYears"].replace(0, 1)

# Manager stability: years with current manager relative to tenure - frequent manager changes
# are a commonly cited attrition driver independent of raw tenure.
df["ManagerStabilityRatio"] = df["YearsWithCurrManager"] / df["YearsAtCompany"].replace(0, 1)

TARGET = "Attrition"
CATEGORICAL = ["BusinessTravel", "Department", "EducationField", "Gender",
               "JobRole", "MaritalStatus", "OverTime"]
# Columns that are pure record-keeping, not predictive signal, or would leak the answer.
DROP = ["EmployeeNumber"]

df = df.drop(columns=DROP)
df_encoded = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)

df_encoded.to_csv("../data/processed/attrition_features.csv", index=False)
print("Feature matrix shape:", df_encoded.shape)
print("Engineered columns added: IncomePerYearAtCompany, PromotionGapRatio, "
      "OverallSatisfaction, ExperienceRatio, ManagerStabilityRatio")
print("Saved -> data/processed/attrition_features.csv")
