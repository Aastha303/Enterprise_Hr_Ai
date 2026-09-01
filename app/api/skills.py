from fastapi import APIRouter, HTTPException
from app.services.skill_gap_service import get_employee_skill_gap

router = APIRouter(prefix="/skills", tags=["skills"])


@router.get("/{employee_id}/gap")
def employee_skill_gap(employee_id: int):
    result = get_employee_skill_gap(employee_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")
    return result
