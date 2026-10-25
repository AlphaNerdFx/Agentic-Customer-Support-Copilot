"""Pydantic request/response models for chat endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, description="UUID session identifier")
    message: str = Field(..., description="Customer message or support query")
    stream: bool = Field(default=False, description="Whether to stream response tokens")


class CitationModel(BaseModel):
    index: int = Field(..., description="Citation marker [N]")
    source_id: str = Field(..., description="KnowledgeArticle source identifier")
    title: str = Field(..., description="Article title")
    excerpt: str = Field(..., description="Verbatim chunk excerpt")
    relevance_score: float = Field(..., description="Vector similarity / rerank score")


class ToolProposalModel(BaseModel):
    tool_call_id: str = Field(default="", description="Unique tool call identifier")
    tool_name: str = Field(..., description="Target tool name")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool invocation parameters")
    rationale: str = Field(..., description="Rationale for proposing the tool action")
    status: str = Field(default="awaiting_approval", description="Gate status")


class ChatResponseContent(BaseModel):
    content: str
    citations: list[CitationModel] = Field(default_factory=list)
    tool_proposal: ToolProposalModel | None = None


class ChatMetricsModel(BaseModel):
    latency_ms: float
    tokens_used: int
    chunks_retrieved: int
    chunks_cited: int


class ChatResponse(BaseModel):
    session_id: str
    response: ChatResponseContent
    metrics: ChatMetricsModel


__all__ = [
    "ChatMetricsModel",
    "ChatRequest",
    "ChatResponse",
    "ChatResponseContent",
    "CitationModel",
    "ToolProposalModel",
]
