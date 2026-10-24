"""Component for rendering citation cards."""
import streamlit as st


def render_citation_card(citation: dict):
    """Render an expandable citation source card."""
    index = citation.get("index", 1)
    title = citation.get("title", "Unknown Source")
    excerpt = citation.get("excerpt", "")
    score = citation.get("relevance_score", 0.0)

    with st.expander(f"[{index}] {title} (Relevance: {score:.2f})"):
        st.write(excerpt)
