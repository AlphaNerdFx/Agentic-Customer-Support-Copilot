"""Tool registry and dispatcher for guarded customer support tools."""

from __future__ import annotations

from typing import Any

from src.agent.tools.email import (
    SendFollowupEmailParams,
    execute_send_followup_email,
)
from src.agent.tools.escalate import (
    EscalateTicketParams,
    execute_escalate_ticket,
)
from src.agent.tools.refund import (
    InitiateRefundParams,
    execute_initiate_refund,
)
from src.core.exceptions import ToolExecutionError

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "escalate_ticket": {
        "name": "escalate_ticket",
        "description": "Escalate high-priority or unresolved customer support disputes to a Tier-2 human specialist.",
        "schema": EscalateTicketParams,
        "executor": execute_escalate_ticket,
    },
    "initiate_refund": {
        "name": "initiate_refund",
        "description": "Initiate a partial or full refund for damaged items, billing overcharges, or return requests.",
        "schema": InitiateRefundParams,
        "executor": execute_initiate_refund,
    },
    "send_followup_email": {
        "name": "send_followup_email",
        "description": "Dispatch a formal resolution summary, invoice copy, or tracking update email to the customer.",
        "schema": SendFollowupEmailParams,
        "executor": execute_send_followup_email,
    },
}


def execute_registered_tool(
    tool_name: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    """Execute a registered tool by name with strict schema validation.

    Args:
        tool_name: Name of the tool to execute.
        parameters: Dictionary of tool invocation arguments.

    Returns:
        Structured tool execution dictionary with status and result summary.

    Raises:
        ToolExecutionError: If the tool is unknown or parameter validation fails.
    """
    if tool_name not in TOOL_REGISTRY:
        raise ToolExecutionError(
            f"Unknown tool '{tool_name}'. Available tools: {list(TOOL_REGISTRY.keys())}"
        )

    tool_entry = TOOL_REGISTRY[tool_name]
    executor = tool_entry["executor"]
    return executor(parameters)


__all__ = [
    "TOOL_REGISTRY",
    "EscalateTicketParams",
    "InitiateRefundParams",
    "SendFollowupEmailParams",
    "execute_escalate_ticket",
    "execute_initiate_refund",
    "execute_registered_tool",
    "execute_send_followup_email",
]
