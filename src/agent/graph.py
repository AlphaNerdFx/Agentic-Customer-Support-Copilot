"""LangGraph state machine for FDE Copilot with HITL approval gates."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.agent.nodes.classify import classify_intent_node
from src.agent.nodes.generate import generate_response_node
from src.agent.nodes.retrieve import retrieve_context_node
from src.agent.nodes.tools import execute_tool_node, propose_tool_node
from src.agent.state import CopilotState
from src.core.exceptions import ApprovalGateError
from src.rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)


def route_tool_decision(
    state: CopilotState,
) -> Literal["no_tool", "needs_tool"]:
    """Conditional router determining whether to trigger human-in-the-loop tool gate."""
    requires_tool = state.get("requires_tool", False)
    proposed_tool = state.get("proposed_tool")
    tool_status = state.get("tool_status", "none")

    if requires_tool and proposed_tool and tool_status != "executed":
        return "needs_tool"
    return "no_tool"


def route_approval_decision(
    state: CopilotState,
) -> Literal["approved", "rejected"]:
    """Conditional router determining whether human approved the proposed action."""
    tool_status = state.get("tool_status", "none")
    if tool_status == "approved":
        return "approved"
    return "rejected"


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    retriever: RAGRetriever | None = None,
) -> Any:
    """Build and compile the LangGraph CopilotState workflow.

    Args:
        checkpointer: Checkpoint saver for state persistence and interrupts.
                      Defaults to in-memory MemorySaver if None is provided.
        retriever: Optional configured RAGRetriever instance.

    Returns:
        Compiled LangGraph state machine runnable.
    """
    graph = StateGraph(CopilotState)

    def retrieve_node(state: CopilotState) -> dict[str, Any]:
        return retrieve_context_node(state, retriever=retriever)

    # Register Nodes
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve_context", retrieve_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("propose_tool", propose_tool_node)
    graph.add_node("execute_tool", execute_tool_node)

    # Register Linear Edges
    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")

    # Register Conditional Edge: Generate -> Propose or END
    graph.add_conditional_edges(
        "generate_response",
        route_tool_decision,
        {
            "no_tool": END,
            "needs_tool": "propose_tool",
        },
    )

    # Register Conditional Edge: Propose -> Execute or END
    graph.add_conditional_edges(
        "propose_tool",
        route_approval_decision,
        {
            "approved": "execute_tool",
            "rejected": END,
        },
    )

    graph.add_edge("execute_tool", END)

    active_checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    return graph.compile(checkpointer=active_checkpointer)


def invoke_chat_turn(
    graph: Any,
    session_id: str,
    message: str,
) -> dict[str, Any]:
    """Invoke a single turn of the copilot conversational graph."""
    config = {"configurable": {"thread_id": session_id}}
    human_msg = HumanMessage(content=message)
    initial_state = {
        "messages": [human_msg],
        "session_id": session_id,
        "retrieved_chunks": [],
        "citations": [],
        "intent": "GENERAL",
        "requires_tool": False,
        "proposed_tool": None,
        "tool_status": "none",
        "tool_result": None,
        "token_count": 0,
        "retrieval_latency_ms": 0.0,
        "generation_latency_ms": 0.0,
    }
    return graph.invoke(initial_state, config=config)


def resume_tool_decision(
    graph: Any,
    session_id: str,
    approved: bool,
    reason: str | None = None,
    modification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resume an interrupted tool approval gate with a human decision.

    Raises:
        ApprovalGateError: If the session is not currently paused at an approval gate.
    """
    config = {"configurable": {"thread_id": session_id}}

    snapshot = graph.get_state(config)
    is_interrupted = bool(snapshot.next and "propose_tool" in snapshot.next)
    if not is_interrupted:
        raise ApprovalGateError(
            f"Cannot resume session '{session_id}': session is not currently in an "
            f"interrupted state awaiting approval (active next: {snapshot.next})."
        )

    resume_payload = {
        "approved": approved,
        "reason": reason,
        "modification": modification,
    }
    return graph.invoke(Command(resume=resume_payload), config=config)


__all__ = [
    "build_graph",
    "invoke_chat_turn",
    "resume_tool_decision",
    "route_approval_decision",
    "route_tool_decision",
]
