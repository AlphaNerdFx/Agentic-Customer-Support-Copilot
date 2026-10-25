"""Component for human-in-the-loop approval gates."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import streamlit as st


def render_approval_card(
    proposal: dict[str, Any],
    session_id: str,
    on_approve: Callable[[dict[str, Any] | None], None],
    on_reject: Callable[[str], None],
) -> None:
    """Render interactive approval card with modification and rejection controls."""
    tool_call_id = proposal.get("tool_call_id", "")
    tool_name = proposal.get("tool_name", "action")
    parameters = proposal.get("parameters", {})
    rationale = proposal.get("rationale", "")
    is_stale = bool(proposal.get("is_stale", False))
    is_expired = bool(proposal.get("is_expired", False))
    elapsed_seconds = float(proposal.get("elapsed_seconds", 0.0))
    hard_ttl_seconds = int(proposal.get("hard_ttl_seconds", 3600))

    if is_expired:
        st.error(
            f"⌛ **Proposal Expired (TTL Exceeded)**: `{tool_name}` proposal has been pending for "
            f"{int(elapsed_seconds)}s (exceeding Hard TTL of {hard_ttl_seconds}s). "
            "Action execution is locked."
        )
        return

    if is_stale:
        st.warning(
            f"⚠️ **Stale Proposal Warning**: This proposal has been awaiting review for "
            f"{int(elapsed_seconds)}s (> 600s). Re-verify customer context before approving."
        )
    else:
        st.warning(f"⚠️ **Action Gate Required**: Copilot proposed `{tool_name}`")

    st.markdown(f"**Supervisor Rationale:** {rationale}")

    # Parameter Inspection & Modification Editor
    with st.expander("🛠️ Inspect / Modify Tool Parameters", expanded=False):
        formatted_json = json.dumps(parameters, indent=2)
        mod_text = st.text_area(
            "Tool Parameters (JSON)",
            value=formatted_json,
            key=f"mod_json_{tool_call_id}",
            height=120,
        )

    # Rejection Reason Input
    rejection_reason = st.text_input(
        "Rejection Reason (required to reject):",
        placeholder="e.g. Need customer order receipt verification first",
        key=f"reject_reason_{tool_call_id}",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Action", key=f"btn_approve_{tool_call_id}", use_container_width=True):
            modification_payload = None
            try:
                parsed_mods = json.loads(mod_text)
                if parsed_mods != parameters:
                    modification_payload = parsed_mods
            except Exception as exc:  # noqa: BLE001
                st.error(f"Invalid JSON parameters: {exc}")
                return
            on_approve(modification_payload)

    with col2:
        if st.button("❌ Reject Action", key=f"btn_reject_{tool_call_id}", use_container_width=True):
            reason_clean = (rejection_reason or "").strip()
            if not reason_clean:
                st.error("Please provide a rejection reason before rejecting.")
                return
            on_reject(reason_clean)


__all__ = ["render_approval_card"]
