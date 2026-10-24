"""Streamlit frontend entry point for FDE Support Copilot."""
import streamlit as st

st.set_page_config(
    page_title="FDE Support Copilot",
    page_icon="🤖",
    layout="wide"
)

st.title("🛡️ FDE Support Copilot")
st.caption("Citation-grounded assistance with human-in-the-loop approval gates.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

st.info("Copilot environment initialized. Ingestion and orchestration pipelines are in progress.")
