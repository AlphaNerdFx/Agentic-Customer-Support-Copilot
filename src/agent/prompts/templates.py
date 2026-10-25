"""Prompt templates and input sanitization guardrails."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.agent.prompts.system import CITATION_SYSTEM_PROMPT

# PII Detection regular expressions (13-19 digits covering standard & extended card numbers)
CREDIT_CARD_PATTERN = re.compile(r"(?<![A-Za-z#\-])\b(?:\d[ -]*?){13,19}\b")
SSN_PATTERN = re.compile(r"(?<![A-Za-z#\-])\b\d{3}[ -]\d{2}[ -]\d{4}\b")

# Adversarial prompt injection keywords and phrases
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions|rules|guidelines)",
    r"disregard\s+(?:all\s+)?(?:previous\s+)?(?:instructions|rules)",
    r"system\s+prompt",
    r"you\s+are\s+now\s+(?:dan|unrestricted)",
    r"dan\s+mode",
    r"jailbreak",
    r"bypass\s+(?:approval|human|guardrails|safety)",
    r"forget\s+(?:all\s+)?instructions",
    r"override\s+(?:system\s+)?rules",
    r"developer\s+mode",
    r"act\s+as\s+an?\s+unrestricted",
]

COMPILED_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS
]

SAFE_INJECTION_REFUSAL = (
    "I cannot fulfill this request. I am programmed to follow strict safety "
    "protocols and cannot bypass system instructions or approvals."
)


def sanitize_pii(text: str) -> tuple[str, list[str]]:
    """Detect and redact credit card numbers and SSNs from text.

    Returns:
        tuple[str, list[str]]: (sanitized_text, detected_pii_types)
    """
    if not text:
        return "", []

    detected_types: list[str] = []
    sanitized = text

    if CREDIT_CARD_PATTERN.search(sanitized):
        sanitized = CREDIT_CARD_PATTERN.sub("[REDACTED_CREDIT_CARD]", sanitized)
        detected_types.append("CREDIT_CARD")

    if SSN_PATTERN.search(sanitized):
        sanitized = SSN_PATTERN.sub("[REDACTED_SSN]", sanitized)
        detected_types.append("SSN")

    return sanitized, detected_types


def detect_prompt_injection(text: str) -> bool:
    """Detect potential prompt injection or adversarial jailbreak attempts.

    Returns:
        bool: True if prompt injection indicators are detected, False otherwise.
    """
    if not text:
        return False

    normalized = " ".join(text.strip().lower().split())
    for pattern in COMPILED_INJECTION_PATTERNS:
        if pattern.search(normalized):
            return True

    return False


def format_generation_messages(
    context_str: str,
    user_query: str,
    conversation_history: list[Any] | None = None,
    max_turns: int = 10,
) -> list[BaseMessage]:
    """Format messages for LLM chat completion including citations and history.

    PRD Edge Case: Preserves the most recent 10 turns verbatim (20 messages).
    If conversation history exceeds 10 turns, older turns are trimmed to prevent
    context window overflow.
    """
    system_content = CITATION_SYSTEM_PROMPT.format(context=context_str)
    messages: list[BaseMessage] = [SystemMessage(content=system_content)]

    if conversation_history:
        max_messages = max_turns * 2
        trimmed_history = (
            conversation_history[-max_messages:]
            if len(conversation_history) > max_messages
            else conversation_history
        )
        for msg in trimmed_history:
            if isinstance(msg, BaseMessage):
                messages.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "")
                if role in ("user", "human"):
                    messages.append(HumanMessage(content=content))
                elif role in ("assistant", "ai"):
                    from langchain_core.messages import AIMessage

                    messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_query))
    return messages


__all__ = [
    "SAFE_INJECTION_REFUSAL",
    "detect_prompt_injection",
    "format_generation_messages",
    "sanitize_pii",
]
