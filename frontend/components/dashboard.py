"""Component for adoption and quality metrics dashboard."""
import streamlit as st


def render_metrics_dashboard(metrics: dict):
    """Render top-level KPI metrics."""
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sessions", metrics.get("total_sessions", 0))
    col2.metric("Avg Latency (ms)", f"{metrics.get('avg_response_time_ms', 0):.0f}")
    col3.metric("Citation Rate", f"{metrics.get('citation_rate', 0.0) * 100:.1f}%")
    col4.metric("Tool Approvals", metrics.get("tool_approvals", 0))
