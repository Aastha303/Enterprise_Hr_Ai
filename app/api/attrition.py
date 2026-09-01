from fastapi import APIRouter, HTTPException
from app.validation.employee_schema import EmployeeAttritionInput
from app.services.attrition_service import predict_for_employee

router = APIRouter(prefix="/predict", tags=["attrition"])


@router.post("/attrition")
def predict_attrition_endpoint(employee: EmployeeAttritionInput):
    try:
        return predict_for_employee(employee.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
