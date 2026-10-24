"""API client utilities for Streamlit frontend."""
import os
import httpx
from typing import Any

API_URL = os.getenv("API_URL", "http://localhost:8000")


def send_chat_message(session_id: str | None, message: str) -> dict[str, Any]:
    """Send a customer query to the FastAPI backend."""
    payload = {
        "session_id": session_id,
        "message": message,
        "stream": False
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(f"{API_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json()


def submit_approval(session_id: str, tool_call_id: str, approved: bool, modification: dict | None = None) -> dict[str, Any]:
    """Submit approval decision to resume LangGraph graph."""
    payload = {
        "session_id": session_id,
        "tool_call_id": tool_call_id,
        "approved": approved,
        "modification": modification
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(f"{API_URL}/approve", json=payload)
        response.raise_for_status()
        return response.json()


def submit_rejection(session_id: str, tool_call_id: str, reason: str) -> dict[str, Any]:
    """Submit rejection decision to abort tool execution in LangGraph."""
    payload = {
        "session_id": session_id,
        "tool_call_id": tool_call_id,
        "reason": reason
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(f"{API_URL}/reject", json=payload)
        response.raise_for_status()
        return response.json()


def get_metrics() -> dict[str, Any]:
    """Fetch aggregated copilot metrics."""
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{API_URL}/metrics")
        response.raise_for_status()
        return response.json()
