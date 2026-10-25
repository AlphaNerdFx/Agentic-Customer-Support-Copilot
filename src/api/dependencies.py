"""FastAPI dependency injection providers and state management."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph import build_graph
from src.core.config import settings
from src.rag.chunking import chunk_article, deduplicate_and_enrich
from src.rag.ingest import DEFAULT_BITEXT_SEED_ROWS
from src.rag.retriever import RAGRetriever
from src.rag.store import ChromaVectorStore
from src.services.metrics import MetricsService
from src.services.session import SessionService

logger = logging.getLogger(__name__)

# Cached singletons
_VECTOR_STORE: ChromaVectorStore | None = None
_RETRIEVER: RAGRetriever | None = None
_AGENT_GRAPH: Any = None
_CHECKPOINTER: MemorySaver | None = None
_SESSION_SERVICE: SessionService | None = None
_METRICS_SERVICE: MetricsService | None = None


def get_vector_store() -> ChromaVectorStore:
    """Return initialized ChromaVectorStore, auto-seeding if collection is empty."""
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        store = ChromaVectorStore()
        count = store.count()
        if count == 0:
            logger.info("ChromaDB vector store is empty. Auto-ingesting canonical seed articles...")
            articles = deduplicate_and_enrich(DEFAULT_BITEXT_SEED_ROWS)
            chunks = []
            for art in articles:
                chunks.extend(chunk_article(art))
            store.add_documents(chunks)
            logger.info(f"Auto-ingested {len(chunks)} chunks into vector store.")
        _VECTOR_STORE = store
    return _VECTOR_STORE


def get_retriever() -> RAGRetriever:
    """Return singleton RAGRetriever backed by initialized ChromaVectorStore."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = RAGRetriever(vector_store=get_vector_store())
    return _RETRIEVER


def get_checkpointer() -> MemorySaver:
    """Return shared checkpointer for session state persistence."""
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        _CHECKPOINTER = MemorySaver()
    return _CHECKPOINTER


def get_agent_graph() -> Any:
    """Return compiled LangGraph state machine runnable."""
    global _AGENT_GRAPH
    if _AGENT_GRAPH is None:
        _AGENT_GRAPH = build_graph(
            retriever=get_retriever(),
            checkpointer=get_checkpointer(),
        )
    return _AGENT_GRAPH


def get_session_service() -> SessionService:
    """Return SessionService singleton."""
    global _SESSION_SERVICE
    if _SESSION_SERVICE is None:
        _SESSION_SERVICE = SessionService(db_path=settings.sqlite_db_path)
    return _SESSION_SERVICE


def get_metrics_service() -> MetricsService:
    """Return MetricsService singleton."""
    global _METRICS_SERVICE
    if _METRICS_SERVICE is None:
        _METRICS_SERVICE = MetricsService(db_path=settings.sqlite_db_path)
    return _METRICS_SERVICE


def reset_api_state(
    db_path: str | None = None,
    custom_vector_store: ChromaVectorStore | None = None,
) -> None:
    """Reset singletons for test isolation."""
    global _VECTOR_STORE, _RETRIEVER, _AGENT_GRAPH, _CHECKPOINTER, _SESSION_SERVICE, _METRICS_SERVICE
    _VECTOR_STORE = custom_vector_store
    _RETRIEVER = RAGRetriever(vector_store=custom_vector_store) if custom_vector_store else None
    _CHECKPOINTER = MemorySaver()
    _AGENT_GRAPH = (
        build_graph(retriever=_RETRIEVER, checkpointer=_CHECKPOINTER)
        if _RETRIEVER
        else None
    )
    _SESSION_SERVICE = SessionService(db_path=db_path) if db_path else None
    _METRICS_SERVICE = MetricsService(db_path=db_path) if db_path else None


__all__ = [
    "get_agent_graph",
    "get_checkpointer",
    "get_metrics_service",
    "get_retriever",
    "get_session_service",
    "get_vector_store",
    "reset_api_state",
]
