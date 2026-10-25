"""Component for rendering citation source cards."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_citation_card(citation: dict[str, Any]) -> None:
    """Render an expandable citation source card with relevance metadata."""
    index = citation.get("index", 1)
    title = citation.get("title", "Support Article")
    source_id = citation.get("source_id", "kb-article")
    excerpt = citation.get("excerpt", "")
    score = float(citation.get("relevance_score", citation.get("score", 0.0)))

    with st.expander(f"[{index}] {title} (ID: {source_id} | Score: {score:.2f})"):
        st.markdown(f"**Knowledge Excerpt:**\n> {excerpt}")


__all__ = ["render_citation_card"]
