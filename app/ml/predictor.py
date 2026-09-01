"""
Turns one validated employee record into a prediction. Applies the SAME
feature engineering used in notebooks/05_feature_engineering.py, then
one-hot encodes and aligns columns to exactly what the model was trained on
(missing dummy columns filled with 0, unexpected ones dropped) so a single
request can never silently shift the model's input schema.
"""
import pandas as pd
from app.ml.model_loader import get_model, get_expected_columns
from app.utils.logger import logger

CATEGORICAL = ["BusinessTravel", "Department", "EducationField", "Gender",
               "JobRole", "MaritalStatus", "OverTime"]


def engineer_features(record: dict) -> pd.DataFrame:
    df = pd.DataFrame([record])

    df["IncomePerYearAtCompany"] = df["MonthlyIncome"] / df["YearsAtCompany"].replace(0, 1)
    df["PromotionGapRatio"] = df["YearsSinceLastPromotion"] / df["YearsAtCompany"].replace(0, 1)
    df["OverallSatisfaction"] = df[["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"]].mean(axis=1)
    df["ExperienceRatio"] = df["YearsAtCompany"] / df["TotalWorkingYears"].replace(0, 1)
    df["ManagerStabilityRatio"] = df["YearsWithCurrManager"] / df["YearsAtCompany"].replace(0, 1)

    df = df.drop(columns=["EmployeeNumber"], errors="ignore")
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)

    expected_cols = get_expected_columns()
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[expected_cols]
    return df_encoded


def risk_bucket(prob: float) -> str:
    if prob >= 0.6:
        return "HIGH"
    if prob >= 0.3:
        return "MEDIUM"
    return "LOW"


def predict_attrition(record: dict) -> dict:
    model = get_model()
    X = engineer_features(record)
    prob = float(model.predict_proba(X)[0, 1])
    risk = risk_bucket(prob)
    logger.info(f"Prediction completed for EmployeeNumber={record.get('EmployeeNumber')} "
                f"prob={prob:.3f} risk={risk}")
    return {"attrition_probability": round(prob, 3), "risk_level": risk}
