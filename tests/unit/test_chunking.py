"""Unit tests for knowledge base category normalization, deduplication, and chunking."""

from __future__ import annotations

import uuid

import pytest

from src.rag.chunking import (
    EXCLUDED_CATEGORIES,
    VALID_CATEGORIES,
    chunk_article,
    deduplicate_and_enrich,
    normalize_category,
)


def test_valid_and_excluded_categories_constants():
    """Verify operational and excluded category tuples."""
    assert "BILLING" in VALID_CATEGORIES
    assert "SHIPPING" in VALID_CATEGORIES
    assert "CANCELLATION" in VALID_CATEGORIES
    assert "ORDER" in VALID_CATEGORIES
    assert "ACCOUNT" in VALID_CATEGORIES
    assert "REFUND" in VALID_CATEGORIES
    assert len(VALID_CATEGORIES) == 6

    assert "FEEDBACK" in EXCLUDED_CATEGORIES
    assert "NEWSLETTER" in EXCLUDED_CATEGORIES


@pytest.mark.parametrize(
    "raw_category,intent,expected",
    [
        ("PAYMENT", "payment_issue", "BILLING"),
        ("INVOICE", "check_invoice", "BILLING"),
        ("billing", "check_invoice", "BILLING"),
        ("DELIVERY", "delivery_options", "SHIPPING"),
        ("SHIPPING_ADDRESS", "change_shipping_address", "SHIPPING"),
        ("shipping", "track_shipment", "SHIPPING"),
        ("CANCELLATION_FEE", "check_cancellation_fee", "CANCELLATION"),
        ("ORDER", "cancel_order", "CANCELLATION"),
        ("order", "cancel_order", "CANCELLATION"),
        ("ORDER", "place_order", "ORDER"),
        ("ORDER", "change_order", "ORDER"),
        ("ORDER", "track_order", "ORDER"),
        ("ACCOUNT", "recover_password", "ACCOUNT"),
        ("account", "create_account", "ACCOUNT"),
        ("REFUND", "track_refund", "REFUND"),
        ("refund", "refund_damaged_item", "REFUND"),
        ("FEEDBACK", "give_feedback", None),
        ("NEWSLETTER", "subscribe_newsletter", None),
        ("UNKNOWN_CAT", "some_intent", None),
        ("", "some_intent", None),
        (None, "some_intent", None),
    ],
)
def test_normalize_category_mappings(
    raw_category: str | None, intent: str, expected: str | None
):
    """Verify all category normalization rules from PRD and SPEC."""
    assert normalize_category(raw_category, intent) == expected


def test_deduplicate_and_enrich():
    """Verify deduplication groups rows, enriches queries, and filters out excluded categories."""
    raw_rows = [
        {
            "category": "REFUND",
            "intent": "track_refund",
            "instruction": "Where is my refund?",
            "response": "Refunds are processed within 3-5 business days.",
        },
        {
            "category": "REFUND",
            "intent": "track_refund",
            "instruction": "How do I check my refund status?",
            "response": "Refunds are processed within 3-5 business days.",
        },
        {
            "category": "REFUND",
            "intent": "track_refund",
            "instruction": "Where is my refund?",  # duplicate query
            "response": "Refunds are processed within 3-5 business days.",
        },
        {
            "category": "PAYMENT",
            "intent": "payment_issue",
            "instruction": "Card declined.",
            "response": "Please check your billing zip code.",
        },
        {
            "category": "FEEDBACK",
            "intent": "feedback_intent",
            "instruction": "Great service!",
            "response": "Thank you.",
        },
    ]

    articles = deduplicate_and_enrich(raw_rows)

    # FEEDBACK should be dropped, leaving 2 articles (REFUND and BILLING)
    assert len(articles) == 2

    # Find the REFUND article
    refund_art = next(a for a in articles if a["category"] == "REFUND")
    assert refund_art["source_id"] == "kb-refund-track_refund"
    # Verify id is a valid RFC 4122 UUID per PRD Data Model
    parsed_uuid = uuid.UUID(refund_art["id"])
    assert parsed_uuid.version == 5
    assert refund_art["title"] == "Track Refund"
    assert "Refunds are processed within 3-5 business days." in refund_art["content"]
    assert (
        refund_art["policy_text"] == "Refunds are processed within 3-5 business days."
    )

    # Verify query deduplication in sample_queries
    sample_queries = refund_art["metadata"]["sample_queries"]
    assert len(sample_queries) == 2
    assert "Where is my refund?" in sample_queries
    assert "How do I check my refund status?" in sample_queries
    assert refund_art["metadata"]["sample_count"] == 3


def test_chunk_article_short():
    """Verify short articles return a single chunk without truncation."""
    article = {
        "id": "kb-order-track_order",
        "source_id": "kb-order-track_order",
        "title": "Track Order",
        "category": "ORDER",
        "content": "Short policy content for tracking orders.",
        "policy_text": "Short policy content for tracking orders.",
        "metadata": {"intent": "track_order", "sample_count": 5},
    }

    chunks = chunk_article(article, max_chars=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "kb-order-track_order-chunk-0"
    assert chunks[0]["source_id"] == "kb-order-track_order"
    assert chunks[0]["article_id"] == "kb-order-track_order"
    assert chunks[0]["category"] == "ORDER"
    assert chunks[0]["content"] == "Short policy content for tracking orders."
    assert chunks[0]["metadata"]["total_chunks"] == 1


def test_chunk_article_long():
    """Verify long articles are properly segmented into overlapping chunks."""
    long_content = (
        "Paragraph 1: Detailed warranty guidelines for returning manufactured goods. "
        * 15
    )
    article = {
        "id": "kb-refund-warranty",
        "source_id": "kb-refund-warranty",
        "title": "Warranty Policy",
        "category": "REFUND",
        "content": long_content,
        "policy_text": "Detailed warranty guidelines for returning manufactured goods.",
        "metadata": {"intent": "warranty_claim"},
    }

    chunks = chunk_article(article, max_chars=250, overlap=50)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_id"] == f"kb-refund-warranty-chunk-{i}"
        assert chunk["source_id"] == "kb-refund-warranty"
        assert chunk["category"] == "REFUND"
        assert chunk["metadata"]["total_chunks"] == len(chunks)
        assert len(chunk["content"]) <= 250
