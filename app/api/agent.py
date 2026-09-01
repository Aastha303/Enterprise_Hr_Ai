from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.agents.orchestrator import route

router = APIRouter(prefix="/agent", tags=["agents"])


class AgentChatRequest(BaseModel):
    message: str
    employee_id: Optional[int] = None
    caller_role: str = "employee"  # employee | manager | hr_admin - see app/agents/tools.py


@router.post("/chat")
def agent_chat(payload: AgentChatRequest):
    return route(payload.message, payload.employee_id, payload.caller_role)
