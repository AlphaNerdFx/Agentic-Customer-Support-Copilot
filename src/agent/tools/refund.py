"""Initiate refund tool for processing monetary customer reimbursements."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.core.exceptions import ToolExecutionError


class InitiateRefundParams(BaseModel):
    """Parameters required to process a refund."""

    reason: str = Field(..., min_length=1, description="Reason for the refund request")
    order_id: str | None = Field(
        default=None, description="Associated order identifier"
    )
    amount: float | None = Field(
        default=None, description="Monetary amount to refund in USD"
    )
    requires_evidence: bool = Field(
        default=False,
        description="Whether customer evidence (e.g., photo of damaged item) is required",
    )

    @field_validator("reason")
    @classmethod
    def validate_reason_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Refund reason cannot be empty")
        return stripped

    @field_validator("amount")
    @classmethod
    def validate_amount_non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("Refund amount cannot be negative")
        return v


def execute_initiate_refund(parameters: dict[str, Any]) -> dict[str, Any]:
    """Execute refund initiation with strict parameter validation.

    Raises:
        ToolExecutionError: If parameters are invalid or amount is negative.
    """
    try:
        params = InitiateRefundParams(**parameters)
    except (ValidationError, TypeError) as exc:
        raise ToolExecutionError(
            f"Invalid parameters for initiate_refund: {exc}"
        ) from exc

    raw_amount = parameters.get("amount")
    if raw_amount is not None:
        try:
            if float(raw_amount) < 0:
                raise ToolExecutionError("Refund amount cannot be negative")
        except (ValueError, TypeError) as exc:
            raise ToolExecutionError(
                f"Invalid amount value: {raw_amount}"
            ) from exc

    content_hash = hashlib.sha256(
        f"{params.reason}:{params.order_id}:{params.amount}:{params.requires_evidence}".encode()
    ).hexdigest()[:8].upper()
    ref = f"RFN-{content_hash}"
    executed_at = datetime.now(UTC).isoformat()

    amount_str = f"${params.amount:.2f}" if params.amount is not None else "full"
    order_str = params.order_id if params.order_id else "unspecified order"

    return {
        "tool_name": "initiate_refund",
        "status": "executed",
        "result": (
            f"Refund request {ref} created successfully for {order_str} "
            f"({amount_str}). Reason: '{params.reason}'. "
            "Customer will be notified via email."
        ),
        "executed_at": executed_at,
    }
