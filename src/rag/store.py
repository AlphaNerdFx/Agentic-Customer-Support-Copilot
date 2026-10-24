"""Vector store abstraction interface and ChromaDB implementation."""
from abc import ABC, abstractmethod
from typing import Any


class VectorStoreService(ABC):
    """Abstract interface for vector retrieval services."""

    @abstractmethod
    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add documents with metadata and embeddings into the store."""
        pass

    @abstractmethod
    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Query top_k most similar document chunks."""
        pass
