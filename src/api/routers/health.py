"""FastAPI health check router reporting infrastructure readiness."""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from src.api.dependencies import get_vector_store
from src.core.config import settings
from src.rag.store import ChromaVectorStore, is_valid_openai_key

router = APIRouter(tags=["Health"])

START_TIME = time.time()


@router.get("/health")
async def health_check(
    store: Annotated[ChromaVectorStore, Depends(get_vector_store)],
) -> dict[str, Any]:
    """Health check endpoint confirming API, ChromaDB vector store, and LLM readiness."""
    try:
        doc_count = store.count()
        vector_status = "connected" if doc_count >= 0 else "degraded"
    except Exception:  # noqa: BLE001
        vector_status = "disconnected"

    llm_provider = (
        "openai" if is_valid_openai_key(settings.openai_api_key) else "deterministic_local"
    )
    uptime_seconds = int(time.time() - START_TIME)

    return {
        "status": "healthy",
        "vector_store": vector_status,
        "llm_provider": llm_provider,
        "uptime_seconds": uptime_seconds,
    }


__all__ = ["router"]
