"""Component for rendering chat history and messages."""
import streamlit as st


def render_chat_history(messages: list[dict]):
    """Render conversation messages in chronological order."""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
