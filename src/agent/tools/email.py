"""Send follow-up email tool for customer communication."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.core.exceptions import ToolExecutionError


class SendFollowupEmailParams(BaseModel):
    """Parameters required to send a follow-up or confirmation email."""

    recipient_email: str = Field(
        default="customer@example.com", description="Destination email address"
    )
    subject: str = Field(..., min_length=1, description="Email subject line")
    body_summary: str = Field(
        ..., min_length=1, description="Summary or body content of the email"
    )
    template_type: str = Field(
        default="resolution_followup", description="Email template identifier"
    )

    @field_validator("recipient_email")
    @classmethod
    def validate_recipient_email(cls, v: str) -> str:
        stripped = v.strip()
        if "@" not in stripped or "." not in stripped:
            raise ValueError(f"Invalid email address: {v}")
        return stripped

    @field_validator("subject", "body_summary")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty")
        return stripped


def execute_send_followup_email(parameters: dict[str, Any]) -> dict[str, Any]:
    """Execute sending follow-up email with parameter validation.

    Raises:
        ToolExecutionError: When parameters fail schema validation.
    """
    try:
        params = SendFollowupEmailParams(**parameters)
    except (ValidationError, TypeError, ValueError) as exc:
        raise ToolExecutionError(
            f"Invalid parameters for send_followup_email: {exc}"
        ) from exc

    content_hash = hashlib.sha256(
        f"{params.recipient_email}:{params.subject}:{params.body_summary}".encode()
    ).hexdigest()[:8].upper()
    ref = f"EML-{content_hash}"
    executed_at = datetime.now(UTC).isoformat()

    return {
        "tool_name": "send_followup_email",
        "status": "executed",
        "result": (
            f"Follow-up email {ref} dispatched to {params.recipient_email} "
            f"with subject '{params.subject}'. Template: '{params.template_type}'."
        ),
        "executed_at": executed_at,
    }
