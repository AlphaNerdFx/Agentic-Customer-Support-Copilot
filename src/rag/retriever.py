"""RAG retrieval service and citation validation helpers."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.rag.store import ChromaVectorStore, VectorStoreService

logger = logging.getLogger(__name__)

# Constants for missing context and fallback messages per SPEC.md & PRD.md
NO_RELEVANT_CONTEXT = "No relevant knowledge base articles found."
INSUFFICIENT_INFO_MESSAGE = (
    "I don't have enough information in our knowledge base to answer this. "
    "Consider checking with a senior agent."
)


def extract_citation_indices(text: str) -> list[int]:
    """Extract unique citation indices like [1], [2] in order of first appearance."""
    seen: set[int] = set()
    indices: list[int] = []
    for match in re.finditer(r"\[(\d+)\]", text):
        idx = int(match.group(1))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def validate_and_clean_citations(
    response_text: str,
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Validate citation indices against retrieved chunks and strip hallucinated references.

    PRD Edge Case: LLM hallucinates a citation index [99] that does not exist in
    the retrieved chunks. Strips out-of-bound citation tags, repairs punctuation spacing,
    and returns the cleaned text along with structured citation objects matching CitationModel.
    """
    if not response_text:
        return "", []

    num_chunks = len(retrieved_chunks)

    def _replace_invalid_tag(match: re.Match) -> str:
        tag_num = int(match.group(1))
        if 1 <= tag_num <= num_chunks:
            return match.group(0)
        return ""

    # Strip invalid / hallucinated tags
    cleaned_text = re.sub(r"\[(\d+)\]", _replace_invalid_tag, response_text)

    # Clean up punctuation artifacts introduced by tag removal (e.g. ', .' or ', ,')
    cleaned_text = re.sub(r",\s*,+", ",", cleaned_text)
    cleaned_text = re.sub(r",\s*\.", ".", cleaned_text)
    cleaned_text = re.sub(r"\s+([,.:;?!])", r"\1", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text).strip()

    # Extract preserved valid citation indices in order of appearance
    valid_indices = extract_citation_indices(cleaned_text)

    citations: list[dict[str, Any]] = []
    for idx in valid_indices:
        chunk = retrieved_chunks[idx - 1]
        source_id = str(chunk.get("source_id", ""))
        title = str(chunk.get("title", "Support Policy"))
        content = str(chunk.get("content", ""))
        excerpt = str(chunk.get("excerpt", content[:280])).strip()
        relevance_score = float(chunk.get("relevance_score", chunk.get("score", 0.0)))

        citations.append(
            {
                "index": idx,
                "source_id": source_id,
                "title": title,
                "excerpt": excerpt,
                "score": relevance_score,
                "relevance_score": relevance_score,
            }
        )

    return cleaned_text, citations


class RAGRetriever:
    """Retriever service querying knowledge base vector store and assembling context."""

    def __init__(self, vector_store: VectorStoreService | None = None) -> None:
        self.vector_store = vector_store or ChromaVectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.15,
        category_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top_k chunks matching the query filtered by minimum relevance score."""
        if not query or not query.strip():
            return []

        hits = self.vector_store.query(
            query_text=query.strip(),
            top_k=top_k,
            category_filter=category_filter,
        )

        # Filter out chunks below minimum relevance threshold
        filtered_hits = [
            hit
            for hit in hits
            if float(hit.get("relevance_score", hit.get("score", 0.0))) >= min_score
        ]
        return filtered_hits

    def format_context_for_llm(self, chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into numbered context blocks for prompt injection.

        Follows SPEC.md Section 6:
        [1] Title (source_id)
        Content: ...
        """
        if not chunks:
            return NO_RELEVANT_CONTEXT

        formatted_blocks: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            title = chunk.get("title", "Knowledge Article")
            source_id = chunk.get("source_id", f"src-{i}")
            content = chunk.get("content", "").strip()

            block = f"[{i}] {title} ({source_id})\n{content}"
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)
