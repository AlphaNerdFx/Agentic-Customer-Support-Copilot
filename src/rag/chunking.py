"""Chunking, category normalization, and query-enriched deduplication for RAG."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Any

# Supported target operational categories
VALID_CATEGORIES: tuple[str, ...] = (
    "BILLING",
    "SHIPPING",
    "CANCELLATION",
    "ORDER",
    "ACCOUNT",
    "REFUND",
)

# Explicitly excluded categories
EXCLUDED_CATEGORIES: tuple[str, ...] = (
    "FEEDBACK",
    "NEWSLETTER",
)


def normalize_category(
    raw_category: str | None, intent: str | None = None
) -> str | None:
    """Normalize raw dataset categories and intents into the 6 target domains.

    Returns None for excluded or unrecognized categories.

    Mapping rules:
      - PAYMENT, INVOICE -> BILLING
      - DELIVERY, SHIPPING_ADDRESS -> SHIPPING
      - CANCELLATION_FEE or (ORDER with intent == 'cancel_order') -> CANCELLATION
      - ORDER (with intent != 'cancel_order') -> ORDER
      - ACCOUNT -> ACCOUNT
      - REFUND -> REFUND
      - FEEDBACK, NEWSLETTER, or unknown -> None
    """
    if not raw_category:
        return None

    cat_clean = raw_category.strip().upper()
    intent_clean = (intent or "").strip().lower()

    if cat_clean in EXCLUDED_CATEGORIES:
        return None

    # Billing mapping
    if cat_clean in ("BILLING", "PAYMENT", "INVOICE"):
        return "BILLING"

    # Shipping mapping
    if cat_clean in ("SHIPPING", "DELIVERY", "SHIPPING_ADDRESS"):
        return "SHIPPING"

    # Cancellation mapping: CANCELLATION_FEE or ORDER with cancel_order/cancel intent
    if cat_clean == "CANCELLATION_FEE" or (
        cat_clean == "ORDER" and intent_clean in ("cancel_order", "cancel", "cancellation")
    ):
        return "CANCELLATION"
    if cat_clean == "CANCELLATION":
        return "CANCELLATION"

    # Order mapping (non-cancellation order intents)
    if cat_clean == "ORDER":
        return "ORDER"

    # Account mapping
    if cat_clean == "ACCOUNT":
        return "ACCOUNT"

    # Refund mapping
    if cat_clean == "REFUND":
        return "REFUND"

    return None


def _format_intent_title(intent: str) -> str:
    """Generate a clean human-readable title from an intent slug."""
    if not intent:
        return "General Support Policy"
    words = intent.replace("_", " ").replace("-", " ").strip().split()
    return " ".join(word.capitalize() for word in words)


def deduplicate_and_enrich(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate raw customer support rows and consolidate into query-enriched articles.

    Groups rows by (normalized_category, intent), extracts the canonical policy
    resolution, collects distinct user queries into sample_queries, and formats
    structured KnowledgeArticle dictionaries matching PRD.md.
    """
    # Group rows by (normalized_cat, intent)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in rows:
        raw_cat = row.get("category") or row.get("raw_category") or ""
        raw_intent = row.get("intent") or ""
        normalized_cat = normalize_category(raw_cat, raw_intent)
        if normalized_cat is None:
            continue

        intent_key = raw_intent.strip().lower()
        if not intent_key:
            intent_key = "general_inquiry"

        key = (normalized_cat, intent_key)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)

    articles: list[dict[str, Any]] = []

    for (category, intent), matching_rows in sorted(grouped.items()):
        # Extract user query variations
        sample_queries: list[str] = []
        seen_queries: set[str] = set()

        # Track canonical responses and raw categories
        response_counts: dict[str, int] = {}
        raw_cats: set[str] = set()

        for row in matching_rows:
            # Query field extraction
            query_text = (
                row.get("instruction") or row.get("query") or row.get("prompt") or ""
            ).strip()
            if query_text and query_text.lower() not in seen_queries:
                seen_queries.add(query_text.lower())
                sample_queries.append(query_text)

            # Response field extraction
            resp_text = (
                row.get("response")
                or row.get("canonical_response")
                or row.get("completion")
                or ""
            ).strip()
            if resp_text:
                response_counts[resp_text] = response_counts.get(resp_text, 0) + 1

            raw_c = row.get("category") or row.get("raw_category")
            if raw_c:
                raw_cats.add(str(raw_c).strip().upper())

        # Determine canonical policy response: mode response or longest response
        if response_counts:
            canonical_response = max(
                response_counts.keys(),
                key=lambda r: (response_counts[r], len(r)),
            )
        else:
            canonical_response = f"Standard operating procedure for {category.lower()} inquiry ({intent})."

        # Build clean title
        provided_title = matching_rows[0].get("title")
        title = (
            provided_title.strip() if provided_title else _format_intent_title(intent)
        )

        # Deterministic identifier
        clean_intent_slug = re.sub(r"[^a-z0-9]+", "_", intent.lower()).strip("_")
        source_id = f"kb-{category.lower()}-{clean_intent_slug}"
        # RFC 4122 compliant UUID derived deterministically from source_id
        article_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, source_id))

        # Format query-enriched content chunk matching SPEC.md Section 6
        sample_questions_formatted = "\n".join(f"- {q}" for q in sample_queries[:12])
        if not sample_questions_formatted:
            sample_questions_formatted = f"- Inquiries regarding {title.lower()}"

        enriched_content = (
            f"Title: {title}\n"
            f"Category: {category} | Intent: {intent}\n"
            f"Typical Customer Questions:\n"
            f"{sample_questions_formatted}\n\n"
            f"Support Policy / Resolution:\n"
            f"{canonical_response}"
        )

        article = {
            "id": article_uuid,
            "source_id": source_id,
            "title": title,
            "category": category,
            "content": enriched_content,
            "policy_text": canonical_response,
            "created_at": datetime.now(UTC).isoformat(),
            "metadata": {
                "intent": intent,
                "sample_queries": sample_queries,
                "sample_count": len(matching_rows),
                "raw_categories": sorted(raw_cats),
            },
        }
        articles.append(article)

    return articles


def chunk_article(
    article: dict[str, Any],
    max_chars: int = 1500,
    overlap: int = 150,
) -> list[dict[str, Any]]:
    """Split a knowledge article into chunks if necessary while preserving metadata.

    If the article content is within max_chars, a single chunk is returned.
    Otherwise, a sliding window with overlap is applied.
    """
    content = article.get("content", "")
    policy_text = article.get("policy_text", "")
    source_id = article.get("source_id", "kb-unknown")
    article_id = article.get("id", str(uuid.uuid5(uuid.NAMESPACE_DNS, source_id)))
    title = article.get("title", "Support Article")
    category = article.get("category", "GENERAL")
    intent = article.get("metadata", {}).get("intent", article.get("intent", ""))
    metadata = article.get("metadata", {})

    # Excerpt represents a clean representative policy snippet
    default_excerpt = (
        policy_text[:280].strip() if policy_text else content[:280].strip()
    )

    if len(content) <= max_chars:
        return [
            {
                "chunk_id": f"{source_id}-chunk-0",
                "source_id": source_id,
                "article_id": article_id,
                "title": title,
                "category": category,
                "intent": intent,
                "content": content,
                "excerpt": default_excerpt,
                "metadata": {
                    **metadata,
                    "article_id": article_id,
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
            }
        ]

    # Sliding window chunking for long articles
    step = max(1, max_chars - overlap)
    chunks: list[dict[str, Any]] = []
    total_est = (len(content) - overlap + (step - 1)) // step

    start = 0
    idx = 0
    while start < len(content):
        end = min(start + max_chars, len(content))
        chunk_text = content[start:end].strip()

        # Find clean excerpt for this chunk
        chunk_excerpt = (
            chunk_text[:280].strip() if len(chunk_text) > 280 else chunk_text
        )

        chunks.append(
            {
                "chunk_id": f"{source_id}-chunk-{idx}",
                "source_id": source_id,
                "article_id": article_id,
                "title": title,
                "category": category,
                "intent": intent,
                "content": chunk_text,
                "excerpt": chunk_excerpt,
                "metadata": {
                    **metadata,
                    "article_id": article_id,
                    "chunk_index": idx,
                    "total_chunks": total_est,
                },
            }
        )

        if end == len(content):
            break
        start += step
        idx += 1

    # Update total_chunks accurately
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)

    return chunks
