"""
Financial Attrition Cost Exposure Model.

Simple, transparent turnover-cost estimate, not a hidden ML model:

    per-employee exposure = MonthlyIncome * 12 * turnover_cost_multiplier * Attrition_Prob

`turnover_cost_multiplier` represents "replacing this person costs roughly
N x their annual salary" (recruiting, onboarding ramp-up, lost productivity)
- a widely used HR industry rule of thumb, typically cited between 0.5x and
2x annual salary depending on role seniority. It's exposed as a slider in
the UI so leadership can stress-test the estimate rather than trust a single
baked-in number.
"""
import pandas as pd
from app.utils.config import EMPLOYEE_INTELLIGENCE_PATH, ATTRITION_PROCESSED_PATH


def _merged() -> pd.DataFrame:
    intel = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)[
        ["EmployeeNumber", "EmployeeName", "Department", "JobRole", "Attrition_Prob", "Risk"]
    ]
    attr = pd.read_csv(ATTRITION_PROCESSED_PATH)[["EmployeeNumber", "MonthlyIncome"]]
    return intel.merge(attr, on="EmployeeNumber", how="left")


def get_financial_exposure(turnover_cost_multiplier: float = 1.5) -> dict:
    df = _merged()
    df["AnnualSalary"] = df["MonthlyIncome"] * 12
    df["Financial_Exposure"] = (
        df["AnnualSalary"] * turnover_cost_multiplier * df["Attrition_Prob"]
    ).round(2)

    total_exposure = round(float(df["Financial_Exposure"].sum()), 2)
    high_risk_exposure = round(float(df.loc[df["Risk"] == "HIGH", "Financial_Exposure"].sum()), 2)

    by_department = (
        df.groupby("Department")["Financial_Exposure"].sum().round(2).reset_index()
        .sort_values("Financial_Exposure", ascending=False)
        .to_dict(orient="records")
    )

    return {
        "turnover_cost_multiplier": turnover_cost_multiplier,
        "total_projected_cost_exposure": total_exposure,
        "high_risk_exposure_portion": high_risk_exposure,
        "by_department": by_department,
        "employees": df[["EmployeeNumber", "EmployeeName", "Department", "Risk", "Financial_Exposure"]]
        .sort_values("Financial_Exposure", ascending=False)
        .to_dict(orient="records"),
    }
