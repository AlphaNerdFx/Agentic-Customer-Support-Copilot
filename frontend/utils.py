"""API client utilities for Streamlit frontend."""

from __future__ import annotations

import os
from typing import Any

import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000")


def send_chat_message(session_id: str | None, message: str) -> dict[str, Any]:
    """Send a customer query to the FastAPI backend."""
    payload = {
        "session_id": session_id,
        "message": message,
        "stream": False,
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(f"{API_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json()


def submit_approval(
    session_id: str,
    tool_call_id: str,
    approved: bool = True,
    modification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit approval decision to resume LangGraph graph and execute tool."""
    payload = {
        "session_id": session_id,
        "tool_call_id": tool_call_id,
        "approved": approved,
        "modification": modification,
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(f"{API_URL}/approve", json=payload)
        response.raise_for_status()
        return response.json()


def submit_rejection(
    session_id: str, tool_call_id: str, reason: str
) -> dict[str, Any]:
    """Submit rejection decision to abort tool execution in LangGraph."""
    payload = {
        "session_id": session_id,
        "tool_call_id": tool_call_id,
        "reason": reason,
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(f"{API_URL}/reject", json=payload)
        response.raise_for_status()
        return response.json()


def get_metrics() -> dict[str, Any]:
    """Fetch aggregated copilot observability and adoption metrics."""
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{API_URL}/metrics")
        response.raise_for_status()
        return response.json()


def get_session_details(session_id: str) -> dict[str, Any]:
    """Fetch session history and pending approval tools for page persistence."""
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{API_URL}/sessions/{session_id}")
        response.raise_for_status()
        return response.json()


def get_health() -> dict[str, Any]:
    """Fetch backend readiness health status."""
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{API_URL}/health")
        response.raise_for_status()
        return response.json()


__all__ = [
    "API_URL",
    "get_health",
    "get_metrics",
    "get_session_details",
    "send_chat_message",
    "submit_approval",
    "submit_rejection",
]
