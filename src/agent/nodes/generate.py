"""Generation node producing citation-grounded customer support replies."""

from __future__ import annotations

import logging
import time
from typing import Any

import tiktoken
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from src.agent.nodes.classify import extract_user_query
from src.agent.prompts.templates import (
    SAFE_INJECTION_REFUSAL,
    format_generation_messages,
    sanitize_pii,
)
from src.agent.state import CopilotState
from src.core.config import settings
from src.rag.retriever import RAGRetriever, validate_and_clean_citations
from src.rag.store import is_valid_openai_key

logger = logging.getLogger(__name__)

INSUFFICIENT_KNOWLEDGE_MESSAGE = (
    "I don't have enough information in our knowledge base to answer this. "
    "Consider escalating to a senior support specialist."
)

_TIKTOKEN_ENCODER = None


def get_token_encoder():
    """Lazily load tiktoken cl100k_base encoder."""
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is None:
        try:
            _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            _TIKTOKEN_ENCODER = None
    return _TIKTOKEN_ENCODER


def count_tokens(text: str) -> int:
    """Accurately count tokens using cl100k_base with heuristic fallback."""
    if not text:
        return 0
    encoder = get_token_encoder()
    if encoder:
        try:
            return len(encoder.encode(text))
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"tiktoken encoding error: {exc}")
    return max(1, len(text.split()) * 2)


def deterministic_grounded_synthesis(
    retrieved_chunks: list[dict[str, Any]],
    user_query: str,
    requires_tool: bool,
    proposed_tool: dict[str, Any] | None,
) -> str:
    """Synthesize deterministic grounded support reply citing retrieved chunks."""
    top_chunk = retrieved_chunks[0]
    title_1 = top_chunk.get("title", "knowledge article")
    content_1 = top_chunk.get("content", "").strip()

    sentences_1 = [s.strip() for s in content_1.split(".") if s.strip()]
    lead_summary = ". ".join(sentences_1[:2]) if sentences_1 else content_1[:200]
    lead_summary = lead_summary.rstrip(".")

    reply_parts: list[str] = [
        f"According to {title_1}, {lead_summary} [1]."
    ]

    if (
        len(retrieved_chunks) > 1
        and float(retrieved_chunks[1].get("relevance_score", 0.0)) >= 0.22
    ):
        second_chunk = retrieved_chunks[1]
        title_2 = second_chunk.get("title", "support documentation")
        content_2 = second_chunk.get("content", "").strip()
        sentences_2 = [s.strip() for s in content_2.split(".") if s.strip()]
        support_info = sentences_2[0] if sentences_2 else content_2[:150]
        support_info = support_info.rstrip(".")
        reply_parts.append(f"Additionally, {title_2} specifies that {support_info} [2].")

    if requires_tool and proposed_tool:
        tool_name = proposed_tool.get("tool_name", "")
        if tool_name == "initiate_refund":
            reply_parts.append(
                "I have drafted a refund request for your review and approval before execution."
            )
        elif tool_name == "escalate_ticket":
            reply_parts.append(
                "I have prepared an escalation ticket for a Tier-2 specialist to investigate."
            )
        elif tool_name == "send_followup_email":
            reply_parts.append(
                "I have prepared a follow-up email draft with these details ready to dispatch."
            )

    return " ".join(reply_parts)


def generate_response_node(state: CopilotState) -> dict[str, Any]:
    """Node: Synthesizes citation-grounded draft reply and updates performance metrics."""
    start_time = time.perf_counter()

    raw_query = extract_user_query(state.get("messages", []))
    sanitized_query, pii_detected = sanitize_pii(raw_query)

    intent = state.get("intent", "GENERAL")
    retrieved_chunks = state.get("retrieved_chunks", [])
    requires_tool = state.get("requires_tool", False)
    proposed_tool = state.get("proposed_tool")

    # 1. Edge Case: Adversarial Prompt Injection
    if intent == "ADVERSARIAL":
        cleaned_text = SAFE_INJECTION_REFUSAL
        citations: list[dict[str, Any]] = []
        token_count = count_tokens(cleaned_text)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "messages": [AIMessage(content=cleaned_text)],
            "citations": citations,
            "token_count": token_count,
            "generation_latency_ms": round(elapsed_ms, 2),
            "tool_status": "none",
            "requires_tool": False,
            "proposed_tool": None,
        }

    # 2. Edge Case: Empty context, out-of-scope query, or low relevance score below threshold 0.08
    max_score = (
        max([float(c.get("relevance_score", c.get("score", 0.0))) for c in retrieved_chunks])
        if retrieved_chunks
        else 0.0
    )

    if not retrieved_chunks or max_score < 0.08 or intent == "OUT_OF_SCOPE":
        logger.info(
            f"Insufficient retrieval context (chunks={len(retrieved_chunks)}, max_score={max_score:.3f}). "
            "Returning insufficient information fallback."
        )
        cleaned_text = INSUFFICIENT_KNOWLEDGE_MESSAGE
        citations = []
        token_count = count_tokens(cleaned_text)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        tool_status = "proposed" if (requires_tool and proposed_tool) else "none"

        return {
            "messages": [AIMessage(content=cleaned_text)],
            "citations": citations,
            "token_count": token_count,
            "generation_latency_ms": round(elapsed_ms, 2),
            "tool_status": tool_status,
        }

    # 3. Grounded Synthesis
    draft_response: str | None = None

    if is_valid_openai_key(settings.openai_api_key):
        try:
            retriever_helper = RAGRetriever()
            context_str = retriever_helper.format_context_for_llm(retrieved_chunks)
            history = state.get("messages", [])[:-1] if state.get("messages") else []
            gen_messages = format_generation_messages(
                context_str=context_str,
                user_query=sanitized_query,
                conversation_history=history,
            )
            llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.1,
            )
            llm_result = llm.invoke(gen_messages)
            if isinstance(llm_result.content, str):
                draft_response = llm_result.content
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Live LLM generation failed ({exc}); using deterministic grounded synthesis."
            )

    if not draft_response:
        draft_response = deterministic_grounded_synthesis(
            retrieved_chunks=retrieved_chunks,
            user_query=sanitized_query,
            requires_tool=requires_tool,
            proposed_tool=proposed_tool,
        )

    # Prepend PII sanitization notice if sensitive info was stripped
    prefix_note = ""
    if pii_detected:
        prefix_note = (
            "Notice: Sensitive personal information (e.g. card/SSN) in your message "
            "was safely redacted before processing.\n\n"
        )

    full_draft = f"{prefix_note}{draft_response}"

    # 4. Strict Citation Post-Processing: Strip Hallucinated Citation Indices
    cleaned_text, citations = validate_and_clean_citations(full_draft, retrieved_chunks)

    token_count = count_tokens(cleaned_text)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    tool_status = "proposed" if (requires_tool and proposed_tool) else "none"

    return {
        "messages": [AIMessage(content=cleaned_text)],
        "citations": citations,
        "token_count": token_count,
        "generation_latency_ms": round(elapsed_ms, 2),
        "tool_status": tool_status,
    }


__all__ = ["generate_response_node"]
