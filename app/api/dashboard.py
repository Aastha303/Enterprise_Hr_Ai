from fastapi import APIRouter, HTTPException
from app.services.engagement_service import get_dashboard_summary, get_attrition_by_department
from app.services.skill_gap_service import get_organization_skill_gaps, get_employee_skill_gap
from app.services.recommendation_service import (
    get_all_recommendations, get_full_employee_record, list_employees, get_raw_employee_record,
    get_recommendation_quality_metrics,
)
from app.services.financial_service import get_financial_exposure
from app.rag.rag_service import get_rag_quality_metrics

router = APIRouter(tags=["dashboard"])


@router.get("/employees")
def employees_roster():
    """Lightweight roster (ID + Name + Dept + Role + Risk) for frontend pickers/search."""
    return list_employees()


@router.get("/employees/{employee_id}/raw")
def employee_raw_record(employee_id: int):
    """Full raw attrition-model input fields for one employee - baseline for the What-If simulator."""
    record = get_raw_employee_record(employee_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")
    return record


@router.get("/dashboard/financial-exposure")
def financial_exposure(turnover_cost_multiplier: float = 1.5):
    return get_financial_exposure(turnover_cost_multiplier)


@router.get("/dashboard/summary")
def dashboard_summary():
    return get_dashboard_summary()


@router.get("/dashboard/attrition-by-department")
def attrition_by_department():
    return get_attrition_by_department()


@router.get("/dashboard/skill-gaps")
def dashboard_skill_gaps():
    return get_organization_skill_gaps()


@router.get("/dashboard/recommendations")
def dashboard_recommendations():
    return get_all_recommendations()


@router.get("/employees/{employee_id}")
def employee_full_record(employee_id: int):
    record = get_full_employee_record(employee_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")
    return record


@router.get("/recommendation-quality")
def recommendation_quality_metrics():
    """
    Recommendation quality evaluation metrics (Precision@k, Recall@k, MRR@k)
    
    Returns comprehensive ranking metrics for skill gap recommendations across all employees:
    - Precision@k: % of top-k recommendations that are correct
    - Recall@k: % of actual skill gaps captured in top-k recommendations
    - MRR@k: Mean Reciprocal Rank (position of first correct recommendation)
    """
    return get_recommendation_quality_metrics()


@router.get("/rag-quality")
def rag_quality_metrics():
    """
    RAG retrieval quality evaluation metrics (Precision@k, Recall@k, MRR@k)
    
    Returns ranking metrics for policy document retrieval quality:
    - Precision@k: % of top-k retrieved documents that are relevant
    - Recall@k: % of relevant documents captured in top-k retrievals
    - MRR@k: Mean Reciprocal Rank (position of first relevant document)
    """
    return get_rag_quality_metrics()
