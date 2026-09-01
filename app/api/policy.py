from fastapi import APIRouter
from pydantic import BaseModel
from app.rag.generator import answer_policy_question

router = APIRouter(prefix="/policy", tags=["policy-rag"])


class PolicyQuestion(BaseModel):
    question: str


@router.post("/ask")
def ask_policy_question(payload: PolicyQuestion):
    return answer_policy_question(payload.question)
