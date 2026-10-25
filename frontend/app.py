"""Streamlit frontend for FDE Support Copilot with interactive HITL approval."""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from frontend.components.approval import render_approval_card
from frontend.components.chat import render_chat_history
from frontend.components.citations import render_citation_card
from frontend.components.dashboard import render_metrics_dashboard
from frontend.utils import (
    get_health,
    get_metrics,
    get_session_details,
    send_chat_message,
    submit_approval,
    submit_rejection,
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="FDE Support Copilot",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "session_id" not in st.session_state or not st.session_state.session_id:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_tool" not in st.session_state:
    st.session_state.pending_tool = None


def load_session_state(sid: str) -> None:
    """Fetch session history and pending approval tools from backend."""
    try:
        details = get_session_details(sid)
        st.session_state.session_id = details.get("session_id", sid)
        st.session_state.messages = details.get("messages", [])
        st.session_state.pending_tool = details.get("pending_tool")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load session details: {exc}")


# ---------------------------------------------------------
# Sidebar: System Health, Metrics, and Session Management
# ---------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Copilot Control")

    # Backend Health Check
    try:
        health = get_health()
        status_color = "green" if health.get("status") == "healthy" else "orange"
        st.markdown(
            f"**System Status:** :{status_color}[● {health.get('status', 'unknown').upper()}]  \n"
            f"**Vector Store:** `{health.get('vector_store', 'unknown')}`  \n"
            f"**LLM Provider:** `{health.get('llm_provider', 'unknown')}`"
        )
    except Exception:  # noqa: BLE001
        st.markdown("**System Status:** :red[● DISCONNECTED]")

    st.divider()

    # Session ID Display & Control
    st.subheader("Conversation Session")
    curr_sid = st.session_state.session_id or "New (unassigned)"
    st.code(curr_sid, language="text")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.pending_tool = None
            st.rerun()

    with col_s2:
        if st.session_state.session_id and st.button(
            "📥 Reload", use_container_width=True
        ):
            load_session_state(st.session_state.session_id)
            st.rerun()

    st.divider()

    # Adoption & Observability Metrics
    st.subheader("Adoption & Quality Metrics")
    try:
        metrics_data = get_metrics()
        render_metrics_dashboard(metrics_data)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Unable to load metrics: {exc}")

    st.divider()

    # Quick Scenarios
    st.subheader("Support Scenarios")
    sample_queries = [
        "What is your refund policy for purchases?",
        "I received a damaged coffee maker for order #ORD-4567, please refund $59.99.",
        "This delivery delay is unacceptable! Transfer me to a supervisor right now.",
        "Please email me a confirmation copy of the invoice to billing-user@example.com.",
    ]
    for q in sample_queries:
        if st.button(f"👉 {q[:38]}...", help=q, key=f"sample_{hash(q)}"):
            st.session_state["quick_prompt"] = q
            st.rerun()


# ---------------------------------------------------------
# Main Panel: Conversation View & HITL Gate
# ---------------------------------------------------------
st.title("🛡️ FDE Support Copilot")
st.caption(
    "Embedded customer support copilot with citation-grounded RAG "
    "and guarded human-in-the-loop tool execution."
)

# Render Chat History
render_chat_history(st.session_state.messages)

# Callback handlers for Approval Gate
def on_tool_approve(modification: dict[str, Any] | None = None) -> None:
    tool_proposal = st.session_state.pending_tool
    if not tool_proposal or not st.session_state.session_id:
        return
    try:
        res = submit_approval(
            session_id=st.session_state.session_id,
            tool_call_id=tool_proposal.get("tool_call_id", ""),
            approved=True,
            modification=modification,
        )
        st.session_state.pending_tool = None
        tool_result = res.get("tool_result", {})
        result_text = tool_result.get("result", "Action executed successfully.")
        st.success(f"Tool executed: {result_text}")
        load_session_state(st.session_state.session_id)
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Approval execution failed: {exc}")


def on_tool_reject(reason: str) -> None:
    tool_proposal = st.session_state.pending_tool
    if not tool_proposal or not st.session_state.session_id:
        return
    try:
        res = submit_rejection(
            session_id=st.session_state.session_id,
            tool_call_id=tool_proposal.get("tool_call_id", ""),
            reason=reason,
        )
        st.session_state.pending_tool = None
        tool_result = res.get("tool_result", {})
        st.info(f"Tool rejected: {tool_result.get('result', reason)}")
        load_session_state(st.session_state.session_id)
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Rejection failed: {exc}")


# Render Pending Approval Card if present
if st.session_state.pending_tool:
    render_approval_card(
        proposal=st.session_state.pending_tool,
        session_id=st.session_state.session_id or "",
        on_approve=on_tool_approve,
        on_reject=on_tool_reject,
    )

# ---------------------------------------------------------
# Chat Input Dispatch
# ---------------------------------------------------------
quick_prompt = st.session_state.pop("quick_prompt", None)
user_input = st.chat_input("Enter customer inquiry or instruction...") or quick_prompt

if user_input:
    # Append user turn to local UI immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Invoke Backend API
    with (
        st.chat_message("assistant"),
        st.spinner("Retrieving knowledge base and drafting grounded reply..."),
    ):
        try:
            response_data = send_chat_message(
                session_id=st.session_state.session_id,
                message=user_input,
            )
            st.session_state.session_id = response_data.get("session_id")
            resp_content = response_data.get("response", {})
            assistant_text = resp_content.get("content", "")
            citations = resp_content.get("citations", [])
            tool_proposal = resp_content.get("tool_proposal")

            st.markdown(assistant_text)
            if citations and isinstance(citations, list):
                st.caption("📚 Referenced Knowledge Base Sources:")
                for cit in citations:
                    render_citation_card(cit)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                    "citations": citations,
                }
            )

            if tool_proposal and tool_proposal.get("status") == "awaiting_approval":
                st.session_state.pending_tool = tool_proposal
                st.rerun()

        except Exception as exc:  # noqa: BLE001
            st.error(f"Error communicating with Copilot API: {exc}")
