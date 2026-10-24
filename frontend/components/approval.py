"""Component for human-in-the-loop approval gate."""
import streamlit as st


def render_approval_card(proposal: dict, on_approve, on_reject):
    """Render approval card with action details and decision buttons."""
    tool_name = proposal.get("tool_name", "Unknown Action")
    parameters = proposal.get("parameters", {})
    rationale = proposal.get("rationale", "")

    st.warning(f"⚠️ Action Required: Proposed execution of `{tool_name}`")
    st.markdown(f"**Rationale:** {rationale}")
    st.json(parameters)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Action", key=f"approve_{tool_name}"):
            on_approve()
    with col2:
        if st.button("❌ Reject Action", key=f"reject_{tool_name}"):
            on_reject()
