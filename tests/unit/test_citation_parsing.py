"""Unit tests for inline citation extraction and hallucination cleaning."""

from __future__ import annotations

from src.rag.retriever import (
    extract_citation_indices,
    validate_and_clean_citations,
)


def test_extract_citation_indices_ordering_and_deduplication():
    """Verify citations are extracted in order of first appearance and deduplicated."""
    text = "Refunds take 3 days [2]. You must file photos [1]. Proof is required [2]."
    indices = extract_citation_indices(text)
    assert indices == [2, 1]


def test_extract_citation_indices_empty_and_no_citations():
    """Verify empty text or text without citations returns empty list."""
    assert extract_citation_indices("") == []
    assert extract_citation_indices("No citations present here.") == []


def test_validate_and_clean_citations_all_valid():
    """Verify valid citations within retrieved chunk range are preserved."""
    retrieved_chunks = [
        {"source_id": "kb-refund-01", "title": "Refund Window", "content": "30 days."},
        {
            "source_id": "kb-refund-02",
            "title": "Evidence Required",
            "content": "Photos needed.",
        },
    ]
    response = "You can request a refund within 30 days [1]. Please provide photos [2]."

    cleaned_text, citations = validate_and_clean_citations(response, retrieved_chunks)

    assert cleaned_text == response
    assert len(citations) == 2
    assert citations[0]["index"] == 1
    assert citations[0]["source_id"] == "kb-refund-01"
    assert citations[0]["title"] == "Refund Window"
    assert citations[1]["index"] == 2
    assert citations[1]["source_id"] == "kb-refund-02"


def test_validate_and_clean_citations_strips_hallucinations():
    """PRD Edge Case: Verify hallucinated citations [99], [0] are removed and text cleaned."""
    retrieved_chunks = [
        {"source_id": "kb-refund-01", "title": "Refund Window", "content": "30 days."},
    ]
    # [99] and [0] are out of range (only chunk 1 exists)
    response = (
        "Refunds are granted within 30 days [1]. Contact priority desk [99]. Note [0]."
    )

    cleaned_text, citations = validate_and_clean_citations(response, retrieved_chunks)

    # Hallucinated tags stripped, valid [1] preserved
    assert "[1]" in cleaned_text
    assert "[99]" not in cleaned_text
    assert "[0]" not in cleaned_text
    assert len(citations) == 1
    assert citations[0]["index"] == 1
    assert citations[0]["source_id"] == "kb-refund-01"


def test_validate_and_clean_citations_zero_chunks():
    """Verify all citations are stripped if no chunks were retrieved."""
    response = "This statement erroneously cites [1] and [2]."
    cleaned_text, citations = validate_and_clean_citations(response, [])

    assert "[1]" not in cleaned_text
    assert "[2]" not in cleaned_text
    assert citations == []


def test_validate_and_clean_citations_punctuation_and_repeated_tags():
    """Verify punctuation cleanup, repeated citations, and schema fields."""
    retrieved_chunks = [
        {
            "source_id": "kb-billing-01",
            "title": "Payment Terms",
            "content": "Due upon receipt.",
            "relevance_score": 0.88,
        },
        {
            "source_id": "kb-shipping-02",
            "title": "Express Delivery",
            "content": "Takes 2 days.",
            "relevance_score": 0.75,
        },
    ]

    # Input has:
    # - [1] cited multiple times across sentences
    # - comma before hallucinated tag [99], e.g. '[2], [99].'
    # - hallucinated [0]
    response = (
        "Invoices are due upon receipt [1]. Delivery takes 2 days [2], [99]. "
        "Remember the invoice policy [1]. Check item [0]."
    )

    cleaned_text, citations = validate_and_clean_citations(response, retrieved_chunks)

    # 1. Hallucinated tags stripped
    assert "[99]" not in cleaned_text
    assert "[0]" not in cleaned_text

    # 2. Valid tags preserved in text
    assert "[1]" in cleaned_text
    assert "[2]" in cleaned_text

    # 3. No hanging comma before period (e.g. '[2],.' should be '[2].')
    assert "[2]," not in cleaned_text
    assert "[2]." in cleaned_text

    # 4. Citations list has exactly 2 unique entries in order of appearance
    assert len(citations) == 2
    assert [c["index"] for c in citations] == [1, 2]

    # 5. Schema verification for both score and relevance_score
    for citation in citations:
        assert "index" in citation
        assert "source_id" in citation
        assert "title" in citation
        assert "excerpt" in citation
        assert "score" in citation
        assert "relevance_score" in citation
        assert citation["score"] == citation["relevance_score"]

