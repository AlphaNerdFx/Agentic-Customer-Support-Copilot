"""RAG module providing ingestion, chunking, storage, and retrieval."""

from src.rag.chunking import (
    EXCLUDED_CATEGORIES,
    VALID_CATEGORIES,
    chunk_article,
    deduplicate_and_enrich,
    normalize_category,
)
from src.rag.ingest import DEFAULT_BITEXT_SEED_ROWS, load_bitext_dataset, run_ingestion
from src.rag.retriever import (
    RAGRetriever,
    extract_citation_indices,
    validate_and_clean_citations,
)
from src.rag.store import (
    ChromaVectorStore,
    DeterministicDomainEmbeddingFunction,
    OpenAIEmbeddingAdapter,
    VectorStoreService,
    get_embedding_function,
)

__all__ = [
    "DEFAULT_BITEXT_SEED_ROWS",
    "EXCLUDED_CATEGORIES",
    "VALID_CATEGORIES",
    "ChromaVectorStore",
    "DeterministicDomainEmbeddingFunction",
    "OpenAIEmbeddingAdapter",
    "RAGRetriever",
    "VectorStoreService",
    "chunk_article",
    "deduplicate_and_enrich",
    "extract_citation_indices",
    "get_embedding_function",
    "load_bitext_dataset",
    "normalize_category",
    "run_ingestion",
    "validate_and_clean_citations",
]
