"""Schemas for tool approval, rejection, and execution."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ToolApproveRequest(BaseModel):
    session_id: str = Field(..., description="LangGraph thread_id to resume")
    tool_call_id: str = Field(..., description="Tool call identifier")
    approved: bool = Field(default=True)
    modification: dict[str, Any] | None = Field(default=None)


class ToolRejectRequest(BaseModel):
    session_id: str = Field(..., description="LangGraph thread_id to resume")
    tool_call_id: str = Field(..., description="Tool call identifier")
    reason: str = Field(..., description="Agent rejection rationale")


class ToolExecutionResult(BaseModel):
    tool_name: str
    status: str
    result: str | None = None
    reason: str | None = None
    executed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class ToolApprovalResponse(BaseModel):
    session_id: str
    tool_result: ToolExecutionResult


__all__ = [
    "ToolApprovalResponse",
    "ToolApproveRequest",
    "ToolExecutionResult",
    "ToolRejectRequest",
]
