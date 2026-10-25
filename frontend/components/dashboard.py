"""Component for adoption and quality metrics dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_metrics_dashboard(metrics: dict[str, Any]) -> None:
    """Render top-level KPI metrics in grid format."""
    col1, col2 = st.columns(2)
    col1.metric("Total Sessions", metrics.get("total_sessions", 0))
    col2.metric("Total Queries", metrics.get("total_queries", 0))

    col3, col4 = st.columns(2)
    avg_latency = metrics.get("avg_response_time_ms", 0.0)
    col3.metric("Avg Latency", f"{avg_latency:.0f} ms")
    citation_rate = metrics.get("citation_rate", 0.0)
    col4.metric("Citation Rate", f"{citation_rate * 100:.1f}%")

    col5, col6 = st.columns(2)
    col5.metric("Tool Proposals", metrics.get("tool_proposals", 0))
    col6.metric("Approvals", metrics.get("tool_approvals", 0))

    col7, col8, col9 = st.columns(3)
    col7.metric("Rejections", metrics.get("tool_rejections", 0))
    col8.metric("TTL Expired", metrics.get("tool_expired", 0))
    avg_citations = metrics.get("avg_citations_per_response", 0.0)
    col9.metric("Avg Citations", f"{avg_citations:.2f}")


__all__ = ["render_metrics_dashboard"]
