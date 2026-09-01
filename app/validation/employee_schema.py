"""
Every prediction request is checked against this schema before it touches
any business logic or the ML model. Bad data gets a 422 response and never
reaches the model, instead of quietly producing a garbage prediction.
"""
from pydantic import BaseModel, Field, field_validator


class EmployeeAttritionInput(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    BusinessTravel: str
    DailyRate: int = Field(..., ge=0)
    Department: str
    DistanceFromHome: int = Field(..., ge=0)
    Education: int = Field(..., ge=1, le=5)
    EducationField: str
    EmployeeNumber: int
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    Gender: str
    HourlyRate: int = Field(..., ge=0)
    JobInvolvement: int = Field(..., ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5)
    JobRole: str
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MaritalStatus: str
    MonthlyIncome: int = Field(..., ge=0)
    MonthlyRate: int = Field(..., ge=0)
    NumCompaniesWorked: int = Field(..., ge=0)
    OverTime: str
    PercentSalaryHike: int = Field(..., ge=0)
    PerformanceRating: int = Field(..., ge=1, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    StockOptionLevel: int = Field(..., ge=0)
    TotalWorkingYears: int = Field(..., ge=0)
    TrainingTimesLastYear: int = Field(..., ge=0)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0)
    YearsInCurrentRole: int = Field(..., ge=0)
    YearsSinceLastPromotion: int = Field(..., ge=0)
    YearsWithCurrManager: int = Field(..., ge=0)

    @field_validator("OverTime")
    @classmethod
    def validate_overtime(cls, v):
        if v not in {"Yes", "No"}:
            raise ValueError("OverTime must be 'Yes' or 'No'")
        return v
