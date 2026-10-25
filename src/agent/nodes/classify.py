"""Intent classification and guarded tool proposal node."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.agent.prompts.system import (
    INTENT_CLASSIFICATION_PROMPT,
)
from src.agent.prompts.templates import (
    detect_prompt_injection,
    sanitize_pii,
)
from src.agent.state import CopilotState
from src.core.config import settings
from src.rag.store import is_valid_openai_key

logger = logging.getLogger(__name__)

CANONICAL_INTENTS = {
    "BILLING",
    "SHIPPING",
    "CANCELLATION",
    "ORDER",
    "ACCOUNT",
    "REFUND",
    "OUT_OF_SCOPE",
    "ADVERSARIAL",
}

# Informational query patterns that do NOT trigger tool calls
INFORMATIONAL_REFUND_PATTERNS = [
    r"^(?:what\s+is|what's|how\s+does|explain)\s+(?:the|your)?\s*refund\s+policy",
    r"^what\s+is\s+the\s+return\s+window",
    r"^how\s+many\s+days\s+to\s+return",
    r"^can\s+you\s+explain\s+the\s+refund",
    r"^tell\s+me\s+about\s+(?:the\s+)?refund",
]

# Actionable triggers for initiate_refund
ACTIONABLE_REFUND_PATTERNS = [
    r"(?:i\s+(?:want|need|demand|request)|please|issue|process)\s+(?:a\s+)?(?:full\s+|partial\s+)?refund",
    r"(?:demand|want|need|request)\s+(?:a\s+)?(?:full\s+|partial\s+)?refund",
    r"refund\s+(?:my|this|me|the)\s+(?:order|money|payment|charge|item|purchase)",
    r"(?:broken|damaged|defective|ruined|crushed|shattered)\s+(?:item|product|box|goods).*refund",
    r"refund.*(?:broken|damaged|defective|ruined|crushed|shattered)",
    r"charged\s+twice.*refund",
    r"get\s+(?:my\s+)?money\s+back",
]

# Actionable triggers for escalate_ticket
ACTIONABLE_ESCALATE_PATTERNS = [
    r"(?:speak|talk|transfer|connect)\s+(?:me\s+)?(?:to\s+)?(?:a|an)?\s*(?:senior\s+|tier-2\s+|lead\s+)?(?:supervisor|manager|human|agent|specialist)",
    r"escalat(?:e|ing|ion)",
    r"(?:unresolved|file\s+a)\s+(?:dispute|complaint)",
    r"(?:account\s+(?:hacked|compromised|stolen))",
    r"this\s+is\s+(?:unacceptable|intolerable)",
]

# Actionable triggers for send_followup_email
ACTIONABLE_EMAIL_PATTERNS = [
    r"(?:send|email|forward)\s+(?:me\s+)?(?:a\s+)?(?:written\s+)?(?:summary\s+)?(?:copy|confirmation|update|receipt|invoice|tracking)",
    r"email\s+(?:confirmation|receipt|invoice|summary)",
    r"send\s+(?:a\s+)?follow-?up\s+email",
    r"(?:send|email|forward)\s+.*(?:to|at)\s+[\w\.-]+@[\w\.-]+\.\w+",
    r"email\s+.*(?:instructions|label|details|receipt|invoice)\s+to\s+[\w\.-]+@[\w\.-]+\.\w+",
]


def extract_user_query(messages: list[Any]) -> str:
    """Extract the most recent user text query from state messages."""
    if not messages:
        return ""

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
        if isinstance(msg, dict):
            if msg.get("role") in ("user", "human"):
                return str(msg.get("content", ""))
        elif hasattr(msg, "content") and getattr(msg, "type", "") in ("human", "user"):
            return str(msg.content)

    # Fallback to last message if types are untyped
    last = messages[-1]
    if hasattr(last, "content"):
        return str(last.content)
    if isinstance(last, dict):
        return str(last.get("content", ""))
    return str(last)


def deterministic_classify_intent(text: str) -> str:
    """Classify user intent into canonical categories using domain rules."""
    lower = text.lower()

    # Priority 1: Refund signals
    if any(k in lower for k in ["refund", "reimbursement", "damaged item", "broken item", "money back"]):
        return "REFUND"

    # Priority 2: Cancellation signals
    if any(k in lower for k in ["cancel", "cancellation", "terminat"]):
        return "CANCELLATION"

    # Priority 3: Billing & Payments
    if any(k in lower for k in ["invoice", "receipt", "charge", "charged", "fee", "payment", "credit card", "billing", "promo", "discount", "coupon"]):
        return "BILLING"

    # Priority 4: Shipping & Delivery
    if any(k in lower for k in ["ship", "delivery", "track", "courier", "package", "address", "transit", "carrier"]):
        return "SHIPPING"

    # Priority 5: Account & Security
    if any(k in lower for k in ["password", "login", "reset", "account", "profile", "sign in", "credentials"]):
        return "ACCOUNT"

    # Priority 6: Order Management
    if any(k in lower for k in ["order", "purchase", "items", "cart", "checkout"]):
        return "ORDER"

    return "OUT_OF_SCOPE"


def detect_actionable_tool(
    text: str, intent: str
) -> tuple[bool, dict[str, Any] | None]:
    """Detect whether user query demands an irreversible guarded tool action."""
    lower = text.lower()

    # 1. Check for initiate_refund
    is_purely_informational_refund = any(
        re.search(pat, lower) for pat in INFORMATIONAL_REFUND_PATTERNS
    )
    is_actionable_refund = any(
        re.search(pat, lower) for pat in ACTIONABLE_REFUND_PATTERNS
    )

    if is_actionable_refund and not (is_purely_informational_refund and not ("i want" in lower or "please" in lower)):
        order_match = re.search(r"order\s*#?([A-Za-z0-9\-]+)", text, re.IGNORECASE)
        amount_match = re.search(r"\$\s*(\d+(?:\.\d{1,2})?)", text)

        requires_evidence = any(
            w in lower for w in ["damaged", "broken", "defective", "shattered", "cracked"]
        )
        reason = "Damaged item received" if requires_evidence else "Customer requested refund"
        if "charged twice" in lower or "overcharge" in lower:
            reason = "Billing overcharge"

        return True, {
            "tool_call_id": f"tc-{uuid.uuid4().hex[:8]}",
            "tool_name": "initiate_refund",
            "parameters": {
                "reason": reason,
                "order_id": order_match.group(1) if order_match else None,
                "amount": float(amount_match.group(1)) if amount_match else None,
                "requires_evidence": requires_evidence,
            },
            "rationale": (
                "Customer requested a financial reimbursement/refund which involves "
                "an irreversible monetary transaction."
            ),
        }

    # 2. Check for escalate_ticket
    is_escalation = any(
        re.search(pat, lower) for pat in ACTIONABLE_ESCALATE_PATTERNS
    )
    if is_escalation:
        priority = "urgent" if any(w in lower for w in ["urgent", "immediately", "asap", "emergency", "furious"]) else "high"
        return True, {
            "tool_call_id": f"tc-{uuid.uuid4().hex[:8]}",
            "tool_name": "escalate_ticket",
            "parameters": {
                "ticket_reason": "Customer requested supervisor or Tier-2 escalation",
                "priority": priority,
                "category": intent if intent in CANONICAL_INTENTS else "GENERAL",
                "customer_summary": text[:200],
            },
            "rationale": (
                "Customer requested escalation to a Tier-2 specialist or supervisor "
                "for an unresolved issue."
            ),
        }

    # 3. Check for send_followup_email
    is_email = any(
        re.search(pat, lower) for pat in ACTIONABLE_EMAIL_PATTERNS
    )
    if is_email:
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        recipient = email_match.group(0) if email_match else "customer@example.com"
        template_type = (
            "invoice_copy"
            if "invoice" in lower or "receipt" in lower
            else ("tracking_update" if "track" in lower or "ship" in lower else "resolution_followup")
        )
        return True, {
            "tool_call_id": f"tc-{uuid.uuid4().hex[:8]}",
            "tool_name": "send_followup_email",
            "parameters": {
                "recipient_email": recipient,
                "subject": f"Follow-up regarding your {intent.lower()} inquiry",
                "body_summary": f"Summary of discussion: {text[:150]}",
                "template_type": template_type,
            },
            "rationale": (
                "Customer requested dispatching an outbound email with customer communication "
                "side-effects."
            ),
        }

    return False, None


def classify_intent_node(state: CopilotState) -> dict[str, Any]:
    """Node: Classifies user intent and proposes guarded tools with input sanitization."""
    raw_query = extract_user_query(state.get("messages", []))
    sanitized_query, _pii_types = sanitize_pii(raw_query)

    # Security Edge Case: Adversarial Prompt Injection
    if detect_prompt_injection(raw_query):
        logger.warning(f"Adversarial prompt injection detected in query: {raw_query[:60]}")
        return {
            "intent": "ADVERSARIAL",
            "requires_tool": False,
            "proposed_tool": None,
        }

    # Live LLM classification if real OpenAI key configured
    intent: str | None = None
    if is_valid_openai_key(settings.openai_api_key):
        try:
            llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.0,
            )
            response = llm.invoke(
                f"{INTENT_CLASSIFICATION_PROMPT}\n\nUser message: {sanitized_query}"
            )
            content = response.content
            if isinstance(content, str):
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    category = parsed.get("category", "").upper()
                    if category in CANONICAL_INTENTS:
                        intent = category
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Live LLM intent classification failed ({exc}); falling back to deterministic rules."
            )

    if not intent:
        intent = deterministic_classify_intent(sanitized_query)

    # Detect tool actions (refund, escalate, email)
    requires_tool, proposed_tool = detect_actionable_tool(sanitized_query, intent)

    return {
        "intent": intent,
        "requires_tool": requires_tool,
        "proposed_tool": proposed_tool,
    }


__all__ = ["classify_intent_node"]
