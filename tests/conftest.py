"""Shared test fixtures for FDE Copilot."""
import pytest

@pytest.fixture
def mock_kb_article():
    return {
        "id": "test-uuid-001",
        "title": "Refund Policy Overview",
        "category": "REFUND",
        "content": "Full refunds are available within 30 days of purchase with valid receipt.",
        "source_id": "row-42",
        "metadata": {
            "intent": "check_refund_policy",
            "sample_queries": ["how do I get my money back?", "refund window duration"],
            "sample_count": 50,
            "raw_categories": ["REFUND"]
        }
    }
