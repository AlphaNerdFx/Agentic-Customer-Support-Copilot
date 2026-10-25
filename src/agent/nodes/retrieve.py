"""RAG retrieval node measuring search latency."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.agent.nodes.classify import extract_user_query
from src.agent.prompts.templates import sanitize_pii
from src.agent.state import CopilotState
from src.rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)


def retrieve_context_node(
    state: CopilotState, retriever: RAGRetriever | None = None
) -> dict[str, Any]:
    """Node: Queries vector store for relevant KB chunks and records latency.

    Skips vector retrieval entirely if the state intent is identified as ADVERSARIAL or OUT_OF_SCOPE.
    """
    intent = state.get("intent", "GENERAL")
    if intent in ("ADVERSARIAL", "OUT_OF_SCOPE"):
        return {
            "retrieved_chunks": [],
            "retrieval_latency_ms": 0.0,
        }

    start_time = time.perf_counter()
    raw_query = extract_user_query(state.get("messages", []))
    sanitized_query, _ = sanitize_pii(raw_query)

    active_retriever = retriever or RAGRetriever()

    canonical_categories = {"BILLING", "SHIPPING", "CANCELLATION", "ORDER", "ACCOUNT", "REFUND"}
    category_filter = intent if intent in canonical_categories else None

    try:
        chunks = active_retriever.retrieve(
            query=sanitized_query,
            top_k=5,
            min_score=0.08,
            category_filter=category_filter,
        )
        if not chunks and category_filter:
            # Fallback to unconstrained search if category filter yielded no chunks
            chunks = active_retriever.retrieve(
                query=sanitized_query,
                top_k=5,
                min_score=0.08,
            )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Error during RAG retrieval: {exc}")
        chunks = []

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "retrieved_chunks": chunks,
        "retrieval_latency_ms": round(elapsed_ms, 2),
    }


__all__ = ["retrieve_context_node"]
