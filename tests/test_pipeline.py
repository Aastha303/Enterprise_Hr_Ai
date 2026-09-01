"""
pytest tests covering the pieces most likely to break silently:
missing columns, invalid scores, prediction shape, risk bucketing, skill gap logic, API status codes.
Run with: pytest tests/ -v  (from the project root)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.ml.predictor import risk_bucket, engineer_features
from app.validation.employee_schema import EmployeeAttritionInput
from pydantic import ValidationError

client = TestClient(app)

VALID_EMPLOYEE = {
    "Age": 29, "BusinessTravel": "Travel_Frequently", "DailyRate": 800, "Department": "Sales",
    "DistanceFromHome": 5, "Education": 3, "EducationField": "Marketing", "EmployeeNumber": 9001,
    "EnvironmentSatisfaction": 2, "Gender": "Male", "HourlyRate": 60, "JobInvolvement": 2,
    "JobLevel": 1, "JobRole": "Sales Executive", "JobSatisfaction": 1, "MaritalStatus": "Single",
    "MonthlyIncome": 3000, "MonthlyRate": 15000, "NumCompaniesWorked": 3, "OverTime": "Yes",
    "PercentSalaryHike": 12, "PerformanceRating": 3, "RelationshipSatisfaction": 2, "StockOptionLevel": 0,
    "TotalWorkingYears": 5, "TrainingTimesLastYear": 1, "WorkLifeBalance": 2, "YearsAtCompany": 2,
    "YearsInCurrentRole": 1, "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 1,
}


def test_missing_required_column_is_caught():
    incomplete = {k: v for k, v in VALID_EMPLOYEE.items() if k != "Department"}
    with pytest.raises(ValidationError):
        EmployeeAttritionInput(**incomplete)


def test_invalid_engagement_style_score_is_rejected():
    bad = dict(VALID_EMPLOYEE, JobSatisfaction=99)  # out of the valid 1-4 range
    with pytest.raises(ValidationError):
        EmployeeAttritionInput(**bad)


def test_attrition_prediction_returns_a_real_probability():
    features = engineer_features(VALID_EMPLOYEE)
    from app.ml.model_loader import get_model
    model = get_model()
    prob = model.predict_proba(features)[0, 1]
    assert 0.0 <= prob <= 1.0


@pytest.mark.parametrize("prob,expected", [(0.9, "HIGH"), (0.61, "HIGH"), (0.4, "MEDIUM"), (0.3, "MEDIUM"), (0.1, "LOW")])
def test_risk_level_assigned_correctly_from_probability(prob, expected):
    assert risk_bucket(prob) == expected


def test_skill_gap_calculation_matches_expected_output():
    gaps = pd.read_csv("data/processed/employee_skill_gaps.csv")
    row = gaps[gaps["EmployeeNumber"] == 1].iloc[0]
    assert row["has_count"] + row["gap_count"] == row["required_count"]


def test_api_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200


def test_api_dashboard_summary_returns_200_and_shape():
    r = client.get("/dashboard/summary")
    assert r.status_code == 200
    body = r.json()
    assert "total_employees" in body and "high_risk_employees" in body


def test_api_predict_valid_input_returns_200():
    r = client.post("/predict/attrition", json=VALID_EMPLOYEE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["attrition_probability"] <= 1.0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_api_predict_invalid_input_returns_422():
    r = client.post("/predict/attrition", json={"Age": 5})
    assert r.status_code == 422


def test_api_unknown_employee_returns_404():
    r = client.get("/employees/999999")
    assert r.status_code == 404


# --- RAG policy Q&A ---

def test_policy_ask_returns_relevant_source():
    r = client.post("/policy/ask", json={"question": "How many weeks of parental leave for a primary caregiver?"})
    assert r.status_code == 200
    body = r.json()
    assert "parental_leave.md" in body["sources"]


def test_policy_ask_out_of_scope_question_has_no_sources():
    r = client.post("/policy/ask", json={"question": "xyzzy quantum flibbertigibbet nonsense query"})
    assert r.status_code == 200
    assert r.json()["sources"] == []


def test_policy_ask_vague_question_offers_topic_list_instead_of_dead_end():
    r = client.post("/policy/ask", json={"question": "whats the policy"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "clarification_needed"
    assert "Pto Leave" in body["answer"] or "PTO" in body["answer"].upper()


# --- Career path ---

def test_career_path_readiness_between_0_and_100():
    r = client.get("/career/1/path")
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["readiness_pct"] <= 100.0


def test_career_path_unknown_employee_returns_404():
    r = client.get("/career/999999/path")
    assert r.status_code == 404


# --- Agent orchestration + governance ---

def test_agent_routes_policy_question_to_policy_agent():
    r = client.post("/agent/chat", json={"message": "What is the remote work policy?"})
    assert r.status_code == 200
    assert r.json()["agent"] == "policy_agent"


def test_agent_routes_attrition_question_to_workforce_agent():
    r = client.post("/agent/chat", json={"message": "attrition risk", "employee_id": 1, "caller_role": "manager"})
    assert r.status_code == 200
    assert r.json()["agent"] == "workforce_agent"


def test_agent_denies_employee_access_to_all_salaries():
    r = client.post("/agent/chat", json={"message": "salary of every employee", "caller_role": "employee"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "permission_denied"


def test_agent_allows_hr_admin_access_to_all_salaries():
    r = client.post("/agent/chat", json={"message": "salary of every employee", "caller_role": "hr_admin"})
    assert r.status_code == 200
    assert "result" in r.json()


def test_agent_denies_employee_access_to_attrition_risk():
    r = client.post("/agent/chat", json={"message": "attrition risk", "employee_id": 1, "caller_role": "employee"})
    assert r.status_code == 200
    assert r.json()["status"] == "permission_denied"
