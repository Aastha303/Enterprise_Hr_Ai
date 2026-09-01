from fastapi import APIRouter, HTTPException
from app.services.career_service import get_career_path

router = APIRouter(prefix="/career", tags=["career"])


@router.get("/{employee_id}/path")
def career_path(employee_id: int):
    result = get_career_path(employee_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
