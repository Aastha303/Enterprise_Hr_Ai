"""
Tool registry + governance layer.

Deck principle (slide 15): "The LLM decides which tool is needed - but
authorization and execution remain outside the LLM. An employee cannot
trigger get_all_employee_salary() simply because the model generated that
tool call. Every action passes through a permissions layer independent of
model output."

This is a deliberately simple implementation of that principle: every tool
has a declared permission level, and the orchestrator checks the caller's
role against it BEFORE executing - the intent-detection step (which tool
to call) never doubles as the authorization step.
"""
from dataclasses import dataclass
from typing import Callable

ROLE_HIERARCHY = {"employee": 0, "manager": 1, "hr_admin": 2}


@dataclass
class Tool:
    name: str
    description: str
    required_role: str  # minimum role from ROLE_HIERARCHY
    func: Callable


TOOLS: dict[str, Tool] = {}


def register_tool(name: str, description: str, required_role: str):
    def decorator(func):
        TOOLS[name] = Tool(name=name, description=description, required_role=required_role, func=func)
        return func
    return decorator


class PermissionDenied(Exception):
    pass


def call_tool(tool_name: str, caller_role: str, **kwargs):
    if tool_name not in TOOLS:
        raise KeyError(f"Unknown tool: {tool_name}")
    tool = TOOLS[tool_name]
    if ROLE_HIERARCHY.get(caller_role, -1) < ROLE_HIERARCHY.get(tool.required_role, 99):
        raise PermissionDenied(
            f"Tool '{tool_name}' requires role '{tool.required_role}' or higher; "
            f"caller has role '{caller_role}'."
        )
    return tool.func(**kwargs)
