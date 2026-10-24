"""Schemas for tool approval, rejection, and execution."""
from pydantic import BaseModel, Field
from datetime import datetime


class ToolApproveRequest(BaseModel):
    session_id: str = Field(..., description="LangGraph thread_id to resume")
    tool_call_id: str = Field(..., description="Tool call identifier")
    approved: bool = Field(default=True)
    modification: dict | None = Field(default=None)


class ToolRejectRequest(BaseModel):
    session_id: str = Field(..., description="LangGraph thread_id to resume")
    tool_call_id: str = Field(..., description="Tool call identifier")
    reason: str = Field(..., description="Agent rejection rationale")


class ToolExecutionResult(BaseModel):
    tool_name: str
    status: str
    result: str | None = None
    reason: str | None = None
    executed_at: datetime = Field(default_factory=datetime.utcnow)


class ToolApprovalResponse(BaseModel):
    session_id: str
    tool_result: ToolExecutionResult
