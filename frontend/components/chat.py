"""Component for rendering chat history, citations, and message metadata."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.components.citations import render_citation_card


def render_chat_history(messages: list[dict[str, Any]]) -> None:
    """Render conversation messages, inline citations, and tool call status."""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        citations = msg.get("citations", [])

        with st.chat_message(role):
            st.markdown(content)

            # Render source citations if present
            if citations and isinstance(citations, list):
                st.caption("📚 Referenced Knowledge Base Sources:")
                for cit in citations:
                    render_citation_card(cit)


__all__ = ["render_chat_history"]
