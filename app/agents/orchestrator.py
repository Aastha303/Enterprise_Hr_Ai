"""
Lightweight stand-in for the deck's LangGraph orchestrator (slide 15).
Real LangGraph brings in a much heavier dependency for what's really just
"pick the right tool(s) given a message" at this MVP scale - so this
implements the same routing + governance contract by hand:

  1. Import app.agents.hr_tools so every tool registers itself.
  2. Detect intent from the message (keyword rules; if ANTHROPIC_API_KEY is
     set, an LLM call can be swapped in for smarter intent detection - the
     ROUTING step, never the permission check).
  3. Call the matched tool through call_tool(), which enforces the
     permission layer regardless of how the tool was chosen.

Swap in real LangGraph later without changing the external /agent/chat
contract - callers only see {agent, tool_used, result}.
"""
import re
from app.agents import hr_tools  # noqa: F401 - import registers the tools
from app.agents.tools import call_tool, PermissionDenied

INTENT_RULES = [
    # Policy-related keywords - HIGHEST PRIORITY to avoid mixing with employee risk data
    # Explicit checks: if asking about company policies, benefits, procedures - route to policy_agent
    # NOT workforce_agent (which combines risk scores with the answer)
    (re.compile(r"\b(policy|leave|pto|benefit|remote|work from|expense|travel|parental|vacation|"
                r"sick day|holiday|time off|maternity|paternity|sabbatical|insurance|healthcare|"
                r"retirement|pension|401|rsu|stock|grant|bonus|overtime|flexible|hybrid|office|"
                r"company offer|what does|how do|do we|can i|can employees)\b", re.I), "policy_agent"),
    # upskilling checked before the generic workforce/risk rule, since phrases like
    # "what skills is this employee missing" mention neither "skill gap" nor
    # "missing skill" in that exact order - match "skill(s)" and "missing/gap" independently.
    (re.compile(r"\bskills?\b.*\b(missing|gap)\b|\b(missing|gap)\b.*\bskills?\b"
                r"|\b(skill gap|upskill|course|recommend|learn|training|certification)\b", re.I), "upskilling_agent"),
    # Workforce/risk queries - LOWER priority than policy so "benefits" doesn't trigger this
    (re.compile(r"\b(risk|attrition|leaving|quit|resign|turnover|churn|retention|will (they|this employee|person) leave)\b", re.I), "workforce_agent"),
    (re.compile(r"\b(career|next role|promotion|path|readiness|development|progression)\b", re.I), "career_agent"),
    (re.compile(r"\b(salary|salaries|compensation|pay of every|income|wage|payroll)\b", re.I), "recruitment_agent"),
]


def detect_agent(message: str) -> str:
    for pattern, agent in INTENT_RULES:
        if pattern.search(message):
            return agent
    return "workforce_agent"  # default: general employee-profile lookup


def route(message: str, employee_id: int | None, caller_role: str = "employee") -> dict:
    agent = detect_agent(message)

    try:
        if agent == "policy_agent":
            # Policy queries return ONLY the policy answer - no employee risk data
            # This ensures benefits queries don't expose attrition_probability or risk level
            result = call_tool("ask_policy", caller_role, question=message)
        elif agent == "workforce_agent":
            if employee_id is None:
                return {"agent": agent, "error": "This request needs an employee_id."}
            profile = call_tool("get_employee_profile", caller_role, employee_id=employee_id)
            risk = call_tool("get_attrition_risk", caller_role, employee_id=employee_id)
            result = {**profile, **risk}
        elif agent == "upskilling_agent":
            if employee_id is None:
                return {"agent": agent, "error": "This request needs an employee_id."}
            gap = call_tool("calculate_skill_gap", caller_role, employee_id=employee_id)
            rec = call_tool("recommend_courses", caller_role, employee_id=employee_id)
            result = {**gap, **rec}
        elif agent == "career_agent":
            if employee_id is None:
                return {"agent": agent, "error": "This request needs an employee_id."}
            result = call_tool("generate_learning_plan", caller_role, employee_id=employee_id)
        elif agent == "recruitment_agent":
            result = call_tool("get_all_employee_salary", caller_role)
        else:
            result = {"error": f"Unrecognized agent: {agent}"}
    except PermissionDenied as e:
        return {"agent": agent, "error": str(e), "status": "permission_denied"}

    return {"agent": agent, "result": result}
