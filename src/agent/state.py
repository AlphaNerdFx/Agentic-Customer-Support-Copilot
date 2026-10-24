"""LangGraph state schema for FDE Copilot."""
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages


class CopilotState(TypedDict):
    """Unified state dictionary tracked by LangGraph checkpointer."""

    # Conversation history
    messages: Annotated[list, add_messages]
    session_id: str

    # RAG Context & Citations
    retrieved_chunks: list[dict]       # [{source_id, title, content, score}]
    citations: list[dict]              # [{index, source_id, title, excerpt, score}]

    # Intent Classification
    intent: str                        # billing, order, account, shipping, refund, cancellation
    requires_tool: bool

    # Tool Calling & HITL Approval
    proposed_tool: dict | None         # {tool_name, parameters, rationale}
    tool_status: Literal["none", "proposed", "approved", "rejected", "executed"]
    tool_result: str | None

    # Performance & Observability Metrics
    token_count: int
    retrieval_latency_ms: float
    generation_latency_ms: float
