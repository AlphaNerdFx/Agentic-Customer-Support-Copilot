"""Dataset loader and schema for golden evaluation test cases."""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field


class GoldenTestCase(BaseModel):
    """Pydantic model representing an annotated support evaluation scenario."""

    id: str = Field(..., description="Unique test case identifier")
    query: str = Field(..., description="Customer input text")
    expected_category: str = Field(
        ..., description="Target category: BILLING, SHIPPING, etc."
    )
    expected_source_ids: list[str] = Field(
        default_factory=list, description="Target knowledge base source IDs"
    )
    expected_tool: str | None = Field(
        default=None, description="Expected tool proposal name if actionable"
    )
    expect_refusal: bool = Field(
        default=False,
        description="Whether query should trigger refusal or out-of-scope fallback",
    )
    contains_pii: bool = Field(
        default=False, description="Whether query contains credit card or SSN PII"
    )
    required_keywords: list[str] = Field(
        default_factory=list,
        description="Substrings expected to appear in valid response",
    )


def load_golden_dataset(
    path: str = "data/eval/golden_test_set.json",
) -> list[GoldenTestCase]:
    """Load and validate the golden evaluation dataset from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Golden test set file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    return [GoldenTestCase(**item) for item in data]


__all__ = ["GoldenTestCase", "load_golden_dataset"]
