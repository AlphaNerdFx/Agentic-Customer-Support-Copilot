"""FastAPI application factory for FDE Copilot."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import chat, health, metrics


def create_app() -> FastAPI:
    """Initialize and configure the FastAPI application."""
    app = FastAPI(
        title="FDE Support Copilot API",
        version="0.1.0",
        description=(
            "Backend API for customer support copilot with citation-grounded RAG "
            "and guarded human-in-the-loop tool execution."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(metrics.router)

    return app


app = create_app()

__all__ = ["app", "create_app"]
