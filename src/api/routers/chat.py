"""Chat and Human-in-the-Loop decision endpoints."""

from __future__ import annotations

import contextlib
import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from src.agent.graph import invoke_chat_turn, resume_tool_decision
from src.api.dependencies import (
    get_agent_graph,
    get_metrics_service,
    get_session_service,
)
from src.api.schemas.chat import (
    ChatMetricsModel,
    ChatRequest,
    ChatResponse,
    ChatResponseContent,
    CitationModel,
    ToolProposalModel,
)
from src.api.schemas.tools import (
    ToolApprovalResponse,
    ToolApproveRequest,
    ToolExecutionResult,
    ToolRejectRequest,
)
from src.core.exceptions import ApprovalGateError, ToolExecutionError
from src.services.metrics import MetricsService
from src.services.session import SessionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


def _check_and_expire_stale_gate(
    session_id: str,
    graph: Any,
    session_service: SessionService,
    metrics_service: MetricsService,
    tool_call_id: str | None = None,
) -> tuple[bool, int]:
    """Check if a pending tool proposal has exceeded its Hard TTL, and lazily expire it if so.

    Returns:
        tuple[bool, int]: (is_expired, hard_ttl_seconds)
    """
    pending = session_service.get_pending_tool(session_id)
    if not pending:
        return False, 0

    hard_ttl = int(pending.get("hard_ttl_seconds", 3600))
    if not pending.get("is_expired"):
        return False, hard_ttl

    # Proposal has exceeded Hard TTL: mark as expired in audit log
    resolved_tool_call_id = tool_call_id or pending.get("tool_call_id")
    did_expire = session_service.expire_tool_proposal(
        session_id=session_id,
        tool_call_id=resolved_tool_call_id,
        reason=f"Expired (Hard TTL {hard_ttl}s exceeded)",
    )
    if did_expire:
        metrics_service.record_tool_expiry(session_id)

    # Safely resume the LangGraph execution to unblock the interrupt gate to END
    with contextlib.suppress(Exception):
        resume_tool_decision(
            graph,
            session_id=session_id,
            approved=False,
            reason=f"Expired (Hard TTL {hard_ttl}s exceeded)",
        )

    return True, hard_ttl


@router.post("/chat", response_model=ChatResponse)
async def chat_turn(
    request: ChatRequest,
    graph: Annotated[Any, Depends(get_agent_graph)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    metrics_service: Annotated[MetricsService, Depends(get_metrics_service)],
) -> ChatResponse:
    """Execute a single conversational turn with RAG citation grounding and tool proposal."""
    session_id = session_service.ensure_session(request.session_id)

    # Prevent concurrent turns while graph is paused at an approval gate,
    # unless the pending proposal has exceeded its Hard TTL (lazy expiration).
    config = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)
    if snapshot.next == ("propose_tool",):
        expired, _ = _check_and_expire_stale_gate(
            session_id=session_id,
            graph=graph,
            session_service=session_service,
            metrics_service=metrics_service,
        )
        if not expired:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Session '{session_id}' has an active tool proposal pending approval. "
                    "Please submit an approval or rejection before sending new messages."
                ),
            )

    try:
        turn_output = invoke_chat_turn(
            graph, session_id=session_id, message=request.message
        )
    except Exception as exc:
        logger.error(f"Error during graph execution turn: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Graph turn execution failed: {exc}"
        ) from exc

    # Extract assistant text
    messages = turn_output.get("messages", [])
    assistant_content = ""
    if messages:
        last = messages[-1]
        assistant_content = (
            str(last.content) if hasattr(last, "content") else str(last)
        )

    # Extract citations
    citations_data = turn_output.get("citations", [])
    citation_models: list[CitationModel] = []
    for c in citations_data:
        citation_models.append(
            CitationModel(
                index=int(c.get("index", 1)),
                source_id=str(c.get("source_id", "")),
                title=str(c.get("title", "Policy")),
                excerpt=str(c.get("excerpt", "")),
                relevance_score=float(c.get("relevance_score", c.get("score", 0.0))),
            )
        )

    # Determine if tool proposal is pending approval
    config = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)
    is_paused = snapshot.next == ("propose_tool",)

    tool_proposal_model: ToolProposalModel | None = None
    has_tool_proposal = False

    if is_paused or turn_output.get("tool_status") == "proposed":
        proposed = (
            turn_output.get("proposed_tool")
            or snapshot.values.get("proposed_tool")
            or {}
        )
        tool_call_id = proposed.get("tool_call_id") or f"tc-{uuid.uuid4().hex[:8]}"
        tool_name = str(proposed.get("tool_name", "tool"))
        parameters = dict(proposed.get("parameters", {}))
        rationale = str(proposed.get("rationale", "Human confirmation required"))

        tool_proposal_model = ToolProposalModel(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            parameters=parameters,
            rationale=rationale,
            status="awaiting_approval",
        )
        session_service.log_tool_proposal(
            session_id=session_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            parameters=parameters,
            rationale=rationale,
        )
        has_tool_proposal = True

    # Performance & Observability metrics
    latency_ms = round(
        float(turn_output.get("generation_latency_ms", 0.0))
        + float(turn_output.get("retrieval_latency_ms", 0.0)),
        1,
    )
    tokens_used = int(turn_output.get("token_count", 0))
    chunks_retrieved = len(turn_output.get("retrieved_chunks", []))
    chunks_cited = len(citation_models)

    # Persist in SQLite
    session_service.add_message(
        session_id=session_id,
        role="user",
        content=request.message,
    )
    session_service.add_message(
        session_id=session_id,
        role="assistant",
        content=assistant_content,
        citations=citations_data,
        tool_proposal=tool_proposal_model.model_dump()
        if tool_proposal_model
        else None,
    )
    metrics_service.record_query_turn(
        session_id=session_id,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        chunks_retrieved=chunks_retrieved,
        chunks_cited=chunks_cited,
        tool_proposed=has_tool_proposal,
    )

    return ChatResponse(
        session_id=session_id,
        response=ChatResponseContent(
            content=assistant_content,
            citations=citation_models,
            tool_proposal=tool_proposal_model,
        ),
        metrics=ChatMetricsModel(
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            chunks_retrieved=chunks_retrieved,
            chunks_cited=chunks_cited,
        ),
    )


@router.post("/approve", response_model=ToolApprovalResponse)
async def approve_tool(
    request: ToolApproveRequest,
    graph: Annotated[Any, Depends(get_agent_graph)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    metrics_service: Annotated[MetricsService, Depends(get_metrics_service)],
) -> ToolApprovalResponse:
    """Submit approval decision to resume LangGraph graph and execute action."""
    # Check for Hard TTL expiration before proceeding
    expired, hard_ttl = _check_and_expire_stale_gate(
        session_id=request.session_id,
        graph=graph,
        session_service=session_service,
        metrics_service=metrics_service,
        tool_call_id=request.tool_call_id,
    )
    if expired:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Tool proposal expired after {hard_ttl} seconds (Hard TTL exceeded). "
                "Approval is no longer permitted."
            ),
        )

    config = {"configurable": {"thread_id": request.session_id}}
    snapshot = graph.get_state(config)

    if snapshot.next != ("propose_tool",):
        raise HTTPException(
            status_code=400,
            detail=(
                f"No pending tool approval gate found for session '{request.session_id}'. "
                "The turn has either completed or no tool was proposed."
            ),
        )

    try:
        resumed = resume_tool_decision(
            graph,
            session_id=request.session_id,
            approved=request.approved,
            modification=request.modification,
        )
    except ApprovalGateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ToolExecutionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    tool_status = resumed.get("tool_status", "executed")
    tool_result = resumed.get("tool_result")
    proposed = resumed.get("proposed_tool") or {}
    tool_name = str(proposed.get("tool_name", "unknown_tool"))

    session_service.update_tool_decision(
        tool_call_id=request.tool_call_id,
        status=tool_status,
        result=tool_result,
    )
    metrics_service.record_tool_decision(
        session_id=request.session_id, approved=request.approved
    )

    return ToolApprovalResponse(
        session_id=request.session_id,
        tool_result=ToolExecutionResult(
            tool_name=tool_name,
            status=tool_status,
            result=tool_result,
            executed_at=datetime.now(UTC),
        ),
    )


@router.post("/reject", response_model=ToolApprovalResponse)
async def reject_tool(
    request: ToolRejectRequest,
    graph: Annotated[Any, Depends(get_agent_graph)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    metrics_service: Annotated[MetricsService, Depends(get_metrics_service)],
) -> ToolApprovalResponse:
    """Submit rejection rationale to cancel tool execution and terminate turn."""
    # Check for Hard TTL expiration before proceeding
    expired, hard_ttl = _check_and_expire_stale_gate(
        session_id=request.session_id,
        graph=graph,
        session_service=session_service,
        metrics_service=metrics_service,
        tool_call_id=request.tool_call_id,
    )
    if expired:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Tool proposal expired after {hard_ttl} seconds (Hard TTL exceeded). "
                "Decision has already been finalized."
            ),
        )

    config = {"configurable": {"thread_id": request.session_id}}
    snapshot = graph.get_state(config)

    if snapshot.next != ("propose_tool",):
        raise HTTPException(
            status_code=400,
            detail=(
                f"No pending tool approval gate found for session '{request.session_id}'. "
                "The turn has either completed or no tool was proposed."
            ),
        )

    try:
        resumed = resume_tool_decision(
            graph,
            session_id=request.session_id,
            approved=False,
            reason=request.reason,
        )
    except ApprovalGateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    tool_status = "rejected"
    tool_result = resumed.get("tool_result")
    proposed = resumed.get("proposed_tool") or {}
    tool_name = str(proposed.get("tool_name", "unknown_tool"))

    session_service.update_tool_decision(
        tool_call_id=request.tool_call_id,
        status="rejected",
        reason=request.reason,
        result=tool_result,
    )
    metrics_service.record_tool_decision(
        session_id=request.session_id, approved=False
    )

    return ToolApprovalResponse(
        session_id=request.session_id,
        tool_result=ToolExecutionResult(
            tool_name=tool_name,
            status=tool_status,
            reason=request.reason,
            result=tool_result,
            executed_at=datetime.now(UTC),
        ),
    )


@router.get("/sessions/{session_id}")
async def get_session_details(
    session_id: str,
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> dict[str, Any]:
    """Retrieve full conversation history and pending tools for session persistence."""
    if not session_service.session_exists(session_id):
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        )
    messages = session_service.get_session_history(session_id)
    pending_tool = session_service.get_pending_tool(session_id)

    return {
        "session_id": session_id,
        "messages": messages,
        "pending_tool": pending_tool,
    }


__all__ = ["router"]
