"""Tool proposal interrupt and execution nodes for LangGraph HITL workflow."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import interrupt

from src.agent.state import CopilotState
from src.agent.tools import execute_registered_tool
from src.core.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)


def propose_tool_node(state: CopilotState) -> dict[str, Any]:
    """Proposes a tool action and pauses execution for human approval via interrupt().

    Pauses graph execution, stores checkpoint state, and surfaces the proposed
    action payload to the supervisor or client. Resumes when a Command(resume=...)
    is provided with approval or rejection rationale.
    """
    tool_proposal = state.get("proposed_tool") or {}

    human_decision = interrupt(
        {
            "tool_call_id": tool_proposal.get("tool_call_id"),
            "tool_name": tool_proposal.get("tool_name"),
            "parameters": tool_proposal.get("parameters", {}),
            "rationale": tool_proposal.get("rationale", ""),
            "prompt": "Do you approve this action?",
        }
    )

    if isinstance(human_decision, dict) and human_decision.get("approved") is True:
        # Merge supervisor modifications if provided
        modification = human_decision.get("modification")
        if modification and isinstance(modification, dict):
            updated_params = dict(tool_proposal.get("parameters", {}))
            updated_params.update(modification)
            tool_proposal["parameters"] = updated_params

        return {
            "tool_status": "approved",
            "proposed_tool": tool_proposal,
            "tool_result": None,
        }

    reason = (
        human_decision.get("reason", "No reason provided")
        if isinstance(human_decision, dict)
        else "No reason provided"
    )
    return {
        "tool_status": "rejected",
        "tool_result": f"Rejected: {reason}",
    }


def execute_tool_node(state: CopilotState) -> dict[str, Any]:
    """Executes the approved tool action using the registered tool executor.

    Raises:
        ToolExecutionError: If no tool was proposed or execution fails.
    """
    tool_proposal = state.get("proposed_tool") or {}
    tool_name = tool_proposal.get("tool_name")
    parameters = tool_proposal.get("parameters", {})

    if not tool_name:
        raise ToolExecutionError(
            "execute_tool_node invoked without a valid proposed tool."
        )

    exec_output = execute_registered_tool(tool_name, parameters)

    return {
        "tool_status": "executed",
        "tool_result": exec_output.get("result"),
        "proposed_tool": tool_proposal,
    }


__all__ = ["execute_tool_node", "propose_tool_node"]
