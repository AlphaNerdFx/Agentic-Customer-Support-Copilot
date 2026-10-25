"""Vector store abstraction interface and ChromaDB implementation."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import chromadb
import numpy as np

from src.core.config import settings

logger = logging.getLogger(__name__)

# Key customer support terms given higher feature weight in deterministic local embeddings
DOMAIN_KEYWORD_WEIGHTS: dict[str, float] = {
    "refund": 4.5,
    "damaged": 4.0,
    "return": 3.8,
    "exchange": 3.5,
    "money": 3.0,
    "invoice": 4.2,
    "bill": 4.0,
    "billing": 4.0,
    "charge": 3.8,
    "payment": 4.0,
    "credit": 3.2,
    "card": 3.0,
    "fee": 3.8,
    "cancel": 4.5,
    "cancellation": 4.5,
    "order": 3.8,
    "delivery": 4.0,
    "shipping": 4.0,
    "shipment": 4.0,
    "address": 3.8,
    "track": 4.0,
    "tracking": 4.0,
    "carrier": 3.5,
    "account": 4.0,
    "password": 4.2,
    "login": 3.8,
    "email": 3.2,
    "reset": 3.8,
    "profile": 3.0,
    "receipt": 4.2,
    "download": 3.8,
    "pdf": 3.8,
    "courier": 4.0,
    "lost": 3.8,
    "package": 3.8,
    "promo": 4.2,
    "coupon": 4.2,
    "discount": 4.2,
    "penalty": 4.0,
    "fulfillment": 4.0,
    "reimbursement": 4.2,
    "dispute": 3.8,
    "policy": 3.0,
    "support": 2.5,
    "status": 3.0,
}


class DeterministicDomainEmbeddingFunction(chromadb.EmbeddingFunction[list[str]]):
    """Deterministic, domain-aware embedding function using signed feature hashing.

    Produces normalized dense vectors (default 384-dim) via token hashing,
    character n-grams, and domain-keyword weighting. Compatible with Chroma's
    embedding function interface.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "deterministic_domain_embedding"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(
        config: dict[str, Any],
    ) -> DeterministicDomainEmbeddingFunction:
        return DeterministicDomainEmbeddingFunction(dim=config.get("dim", 384))

    def _embed_single(self, text: str) -> list[float]:
        if not text or not text.strip():
            # Return zero unit vector
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = 1.0
            return vec.tolist()

        vec = np.zeros(self.dim, dtype=np.float32)
        cleaned = text.lower()
        words = re.findall(r"[a-z0-9]+", cleaned)

        # 1. Word token hashing with domain weighting and root canonicalization
        for w in words:
            matched_kw: str | None = None
            boost = 1.0
            for kw, weight in DOMAIN_KEYWORD_WEIGHTS.items():
                if kw in w and weight > boost:
                    boost = weight
                    matched_kw = kw

            # Surface word token hashing
            h = hashlib.sha256(w.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % self.dim
            sign = 1.0 if (int(h[8:10], 16) % 2 == 0) else -1.0
            vec[idx] += sign * 1.0

            # If domain keyword matched, hash canonical root so all inflections align
            if matched_kw:
                h_root = hashlib.sha256(matched_kw.encode("utf-8")).hexdigest()
                idx_root = int(h_root[:8], 16) % self.dim
                sign_root = 1.0 if (int(h_root[8:10], 16) % 2 == 0) else -1.0
                vec[idx_root] += sign_root * boost

            # Stem common plurals (e.g., 'fees' -> 'fee', 'orders' -> 'order')
            if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
                stem = w[:-1]
                h_stem = hashlib.sha256(stem.encode("utf-8")).hexdigest()
                idx_stem = int(h_stem[:8], 16) % self.dim
                sign_stem = 1.0 if (int(h_stem[8:10], 16) % 2 == 0) else -1.0
                vec[idx_stem] += sign_stem * 0.8

        # 2. Bigram hashing for phrase context (e.g., 'cancel order', 'damaged item')
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i + 1]}"
            h = hashlib.sha256(bigram.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % self.dim
            sign = 1.0 if (int(h[8:10], 16) % 2 == 0) else -1.0
            vec[idx] += sign * 1.5

        # 3. Character 3-grams for morphological subwords
        for i in range(len(cleaned) - 2):
            trigram = cleaned[i : i + 3]
            if trigram.strip():
                h = hashlib.sha256(trigram.encode("utf-8")).hexdigest()
                idx = int(h[:8], 16) % self.dim
                sign = 1.0 if (int(h[8:10], 16) % 2 == 0) else -1.0
                vec[idx] += sign * 0.35

        # L2 Normalization
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return vec.tolist()

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Chroma-compatible embedding interface."""
        return [self._embed_single(text) for text in input]


class OpenAIEmbeddingAdapter(chromadb.EmbeddingFunction[list[str]]):
    """Adapter for OpenAI text-embedding-3-small with deterministic fallback."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        fallback_fn: DeterministicDomainEmbeddingFunction | None = None,
    ) -> None:
        import openai

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.fallback_fn = fallback_fn or DeterministicDomainEmbeddingFunction()

    @staticmethod
    def name() -> str:
        return "openai_embedding_adapter"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict[str, Any]:
        return {"model": self.model}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> OpenAIEmbeddingAdapter:
        return OpenAIEmbeddingAdapter(
            api_key="", model=config.get("model", "text-embedding-3-small")
        )

    def __call__(self, input: list[str]) -> list[list[float]]:
        try:
            # Batch embedding via OpenAI API
            response = self.client.embeddings.create(input=input, model=self.model)
            return [record.embedding for record in response.data]
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"OpenAI embedding call failed ({e}). Falling back to deterministic embeddings."
            )
            return self.fallback_fn(input)


def is_valid_openai_key(key: str | None) -> bool:
    """Check if the provided string is a plausible real OpenAI API key."""
    if not key or not isinstance(key, str):
        return False
    k = key.strip()
    k_lower = k.lower()
    if any(token in k_lower for token in ("your-openai", "placeholder", "paste", "example", "test-key")):
        return False
    return k.startswith("sk-") and len(k) >= 25


def get_embedding_function(
    api_key: str | None = None,
    model: str | None = None,
) -> Any:
    """Return configured embedding function: live OpenAI if key is present, else local deterministic."""
    target_key = api_key if api_key is not None else settings.openai_api_key
    target_model = model or settings.embedding_model

    if is_valid_openai_key(target_key):
        logger.info(f"Using live OpenAI embedding model: {target_model}")
        return OpenAIEmbeddingAdapter(api_key=target_key.strip(), model=target_model)

    logger.info(
        "Using deterministic domain embedding function (zero-mock offline mode)"
    )
    return DeterministicDomainEmbeddingFunction()


class VectorStoreService(ABC):
    """Abstract interface for vector retrieval services."""

    @abstractmethod
    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add documents with metadata and embeddings into the store."""

    @abstractmethod
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query top_k most similar document chunks."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of chunks indexed in the store."""

    @abstractmethod
    def reset(self) -> None:
        """Purge and reset the vector store collection."""


class ChromaVectorStore(VectorStoreService):
    """Production-grade ChromaDB vector store implementation with metadata flattening."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = "fde_knowledge_base",
        client: Any | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_function = embedding_function or get_embedding_function()

        if client is not None:
            self.client = client
        elif persist_dir is not None:
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
        elif settings.chroma_persist_dir:
            os.makedirs(settings.chroma_persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        else:
            self.client = chromadb.EphemeralClient()

        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
        except ValueError:
            logger.warning(
                f"Embedding function changed for collection '{self.collection_name}'; recreating collection."
            )
            with contextlib.suppress(Exception):
                self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add or upsert documents with Chroma-compatible flat metadata."""
        if not documents:
            return

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, str | int | float | bool]] = []

        for doc in documents:
            doc_id = str(
                doc.get("chunk_id")
                or doc.get("id")
                or f"{doc.get('source_id', 'doc')}-chunk-0"
            )
            content = str(doc.get("content", "")).strip()
            if not content:
                continue

            metadata_dict = doc.get("metadata", {})
            sample_queries = (
                metadata_dict.get("sample_queries") or doc.get("sample_queries") or []
            )
            raw_cats = (
                metadata_dict.get("raw_categories") or doc.get("raw_categories") or []
            )

            # Chroma metadata values must be str, int, float, or bool
            flat_meta: dict[str, str | int | float | bool] = {
                "source_id": str(doc.get("source_id", doc_id)),
                "title": str(doc.get("title", "Support Article")),
                "category": str(doc.get("category", "")),
                "intent": str(metadata_dict.get("intent") or doc.get("intent") or ""),
                "excerpt": str(doc.get("excerpt", content[:280])),
                "sample_count": int(
                    metadata_dict.get("sample_count", doc.get("sample_count", 1))
                ),
                "sample_queries_json": json.dumps(sample_queries),
                "raw_categories_json": json.dumps(raw_cats),
            }

            ids.append(doc_id)
            texts.append(content)
            metadatas.append(flat_meta)

        if ids:
            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query top_k most similar document chunks, returning normalized relevance scores."""
        total_count = self.collection.count()
        if total_count == 0 or not query_text.strip():
            return []

        n_results = min(top_k, total_count)
        where_filter = (
            {"category": category_filter.strip().upper()} if category_filter else None
        )

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Chroma query execution failed: {e}")
            return []

        hits: list[dict[str, Any]] = []
        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        meta_list = results.get("metadatas", [[]])[0]
        dist_list = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids_list):
            doc_content = docs_list[i] if i < len(docs_list) else ""
            raw_meta = meta_list[i] if i < len(meta_list) else {}
            dist = float(dist_list[i]) if i < len(dist_list) else 0.0

            # Cosine distance d in [0, 2] -> relevance score in [0.0, 1.0]
            relevance = max(0.0, min(1.0, round(1.0 - max(0.0, dist), 4)))

            # Deserialize JSON fields
            sample_queries = []
            if "sample_queries_json" in raw_meta:
                try:
                    sample_queries = json.loads(str(raw_meta["sample_queries_json"]))
                except (json.JSONDecodeError, TypeError):
                    sample_queries = []

            source_id = str(raw_meta.get("source_id", doc_id))
            title = str(raw_meta.get("title", "Support Article"))
            category = str(raw_meta.get("category", ""))
            intent = str(raw_meta.get("intent", ""))
            excerpt = str(raw_meta.get("excerpt", doc_content[:280]))

            hit = {
                "chunk_id": doc_id,
                "source_id": source_id,
                "title": title,
                "content": doc_content,
                "excerpt": excerpt,
                "category": category,
                "intent": intent,
                "score": relevance,
                "relevance_score": relevance,
                "metadata": {
                    "source_id": source_id,
                    "title": title,
                    "category": category,
                    "intent": intent,
                    "sample_count": int(raw_meta.get("sample_count", 1)),
                    "sample_queries": sample_queries,
                },
            }
            hits.append(hit)

        # Ensure sorted descending by relevance score
        hits.sort(key=lambda h: h["relevance_score"], reverse=True)
        return hits

    def count(self) -> int:
        """Return total chunks indexed in collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Reset and recreate the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Collection deletion during reset skipped: {e}")
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
        except ValueError:
            logger.warning(
                f"Embedding function changed for collection '{self.collection_name}'; recreating collection."
            )
            with contextlib.suppress(Exception):
                self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
