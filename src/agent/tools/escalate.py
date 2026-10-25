"""Escalate ticket tool for handling tier-2 customer support handoffs."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from src.core.exceptions import ToolExecutionError


class EscalateTicketParams(BaseModel):
    """Parameters required to escalate an issue to a Tier-2 specialist."""

    ticket_reason: str = Field(..., min_length=1, description="Reason for escalation")
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        default="high", description="Urgency priority level"
    )
    category: str = Field(
        default="GENERAL", description="Support category for escalation routing"
    )
    customer_summary: str = Field(
        default="", description="Summary of customer interactions and sentiment"
    )


def execute_escalate_ticket(parameters: dict[str, Any]) -> dict[str, Any]:
    """Execute ticket escalation with validation and deterministic reference creation.

    Raises:
        ToolExecutionError: When parameters fail schema validation.
    """
    try:
        params = EscalateTicketParams(**parameters)
    except (ValidationError, TypeError) as exc:
        raise ToolExecutionError(
            f"Invalid parameters for escalate_ticket: {exc}"
        ) from exc

    content_hash = hashlib.sha256(
        f"{params.ticket_reason}:{params.priority}:{params.category}:{params.customer_summary}".encode()
    ).hexdigest()[:8].upper()
    ref = f"ESC-{content_hash}"
    executed_at = datetime.now(UTC).isoformat()

    return {
        "tool_name": "escalate_ticket",
        "status": "executed",
        "result": (
            f"Escalation ticket {ref} created with priority '{params.priority}' "
            f"for category '{params.category}'. Assigned to Tier-2 specialist."
        ),
        "executed_at": executed_at,
    }
