"""Unit tests for RAG retriever and real ChromaDB vector store."""

from __future__ import annotations

import chromadb
import pytest

from src.rag.chunking import chunk_article, deduplicate_and_enrich
from src.rag.ingest import DEFAULT_BITEXT_SEED_ROWS
from src.rag.retriever import NO_RELEVANT_CONTEXT, RAGRetriever
from src.rag.store import ChromaVectorStore, DeterministicDomainEmbeddingFunction


@pytest.fixture
def real_ephemeral_store():
    """Create a real ChromaVectorStore backed by an ephemeral in-memory client."""
    client = chromadb.EphemeralClient()
    embed_fn = DeterministicDomainEmbeddingFunction(dim=384)
    store = ChromaVectorStore(
        collection_name="test_knowledge_base",
        client=client,
        embedding_function=embed_fn,
    )
    # Populate real test documents
    docs = [
        {
            "chunk_id": "kb-refund-damaged-chunk-0",
            "source_id": "kb-refund-damaged",
            "title": "Refund Policy for Damaged Items",
            "category": "REFUND",
            "content": (
                "Title: Refund Policy for Damaged Items\n"
                "Category: REFUND | Intent: refund_damaged_item\n"
                "Typical Customer Questions:\n"
                "- My package arrived broken. How do I get a refund?\n"
                "- Defective item received, need reimbursement.\n\n"
                "Support Policy / Resolution:\n"
                "For items that arrive damaged or defective, we provide an immediate 100% refund."
            ),
            "excerpt": "For items that arrive damaged or defective, we provide an immediate 100% refund.",
            "metadata": {
                "intent": "refund_damaged_item",
                "sample_count": 5,
                "sample_queries": ["My package arrived broken."],
            },
        },
        {
            "chunk_id": "kb-shipping-tracking-chunk-0",
            "source_id": "kb-shipping-tracking",
            "title": "Track Carrier Shipment",
            "category": "SHIPPING",
            "content": (
                "Title: Track Carrier Shipment\n"
                "Category: SHIPPING | Intent: track_shipment\n"
                "Typical Customer Questions:\n"
                "- How do I track my package?\n"
                "- Where is the carrier tracking number?\n\n"
                "Support Policy / Resolution:\n"
                "Carrier tracking links for FedEx, UPS, or USPS are emailed when your order dispatches."
            ),
            "excerpt": "Carrier tracking links for FedEx, UPS, or USPS are emailed when your order dispatches.",
            "metadata": {
                "intent": "track_shipment",
                "sample_count": 4,
                "sample_queries": ["Where is my tracking number?"],
            },
        },
        {
            "chunk_id": "kb-billing-invoice-chunk-0",
            "source_id": "kb-billing-invoice",
            "title": "Invoice Download",
            "category": "BILLING",
            "content": (
                "Title: Invoice Download\n"
                "Category: BILLING | Intent: check_invoice\n"
                "Typical Customer Questions:\n"
                "- How do I download my receipt?\n\n"
                "Support Policy / Resolution:\n"
                "Download past tax invoices from the Billing section of your Account Dashboard."
            ),
            "excerpt": "Download past tax invoices from the Billing section.",
            "metadata": {"intent": "check_invoice", "sample_count": 2},
        },
    ]
    store.add_documents(docs)
    return store


def test_real_chroma_store_count_and_query(real_ephemeral_store):
    """Verify documents are indexed into real Chroma and queried with normalized scores."""
    assert real_ephemeral_store.count() == 3

    results = real_ephemeral_store.query(
        "How to get refund for broken damaged item?", top_k=2
    )
    assert len(results) > 0
    top_hit = results[0]

    assert top_hit["category"] == "REFUND"
    assert top_hit["source_id"] == "kb-refund-damaged"
    assert 0.0 <= top_hit["relevance_score"] <= 1.0
    assert "damaged" in top_hit["content"].lower()


def test_real_chroma_category_filter(real_ephemeral_store):
    """Verify category filtering restricts retrieval to specified operational domain."""
    results = real_ephemeral_store.query(
        "Where is my package?",
        top_k=5,
        category_filter="SHIPPING",
    )
    assert len(results) >= 1
    for hit in results:
        assert hit["category"] == "SHIPPING"


def test_retriever_retrieve_and_format_context(real_ephemeral_store):
    """Verify RAGRetriever filters low-scoring chunks and formats prompt context blocks."""
    retriever = RAGRetriever(vector_store=real_ephemeral_store)

    chunks = retriever.retrieve("damaged package refund", top_k=2, min_score=0.10)
    assert len(chunks) >= 1
    assert chunks[0]["category"] == "REFUND"

    formatted_context = retriever.format_context_for_llm(chunks)
    assert (
        "[1] Refund Policy for Damaged Items (kb-refund-damaged)" in formatted_context
    )
    assert "Support Policy / Resolution:" in formatted_context


def test_retriever_empty_query():
    """Verify empty query returns empty list without error."""
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(collection_name="empty_test", client=client)
    retriever = RAGRetriever(vector_store=store)

    assert retriever.retrieve("") == []
    assert (
        retriever.format_context_for_llm([])
        == NO_RELEVANT_CONTEXT
    )


def test_real_chroma_persistence_across_reinstantiation(tmp_path):
    """Zero-Mock: Verify ChromaVectorStore persists data across client re-instantiation."""
    persist_dir = str(tmp_path / "chroma_persist_test")
    embed_fn = DeterministicDomainEmbeddingFunction(dim=384)

    # 1. First instance: write documents
    store1 = ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name="persisted_kb",
        embedding_function=embed_fn,
    )
    docs = [
        {
            "chunk_id": "kb-order-place-chunk-0",
            "source_id": "kb-order-place",
            "title": "Place Online Order",
            "category": "ORDER",
            "content": "Add items to your online cart and proceed to checkout to place an order.",
            "excerpt": "Add items to your online cart to place an order.",
            "metadata": {"intent": "place_order", "sample_count": 3},
        },
        {
            "chunk_id": "kb-cancellation-fee-chunk-0",
            "source_id": "kb-cancellation-fee",
            "title": "Order Cancellation Fee",
            "category": "CANCELLATION",
            "content": "Orders may be cancelled within 30 minutes free of penalty fees.",
            "excerpt": "Orders cancelled within 30 minutes free of charge.",
            "metadata": {"intent": "check_cancellation_fee", "sample_count": 2},
        },
    ]
    store1.add_documents(docs)
    assert store1.count() == 2

    # 2. Second instance: re-instantiate store from the same directory (no documents added)
    store2 = ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name="persisted_kb",
        embedding_function=embed_fn,
    )
    assert store2.count() == 2

    # 3. Query the re-instantiated store
    hits = store2.query("penalty for cancelling an order", top_k=1)
    assert len(hits) == 1
    assert hits[0]["category"] == "CANCELLATION"
    assert hits[0]["source_id"] == "kb-cancellation-fee"
    assert hits[0]["relevance_score"] > 0.0


def test_top1_retrieval_across_all_six_categories(tmp_path):
    """Zero-Mock: Verify Top-1 retrieval accuracy across all 6 operational categories."""
    persist_dir = str(tmp_path / "all_cats_kb")
    embed_fn = DeterministicDomainEmbeddingFunction(dim=384)
    store = ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name="all_cats_collection",
        embedding_function=embed_fn,
    )

    # Ingest entire seed corpus (20 articles across 6 categories)
    articles = deduplicate_and_enrich(DEFAULT_BITEXT_SEED_ROWS)
    chunks = []
    for art in articles:
        chunks.extend(chunk_article(art))
    store.add_documents(chunks)
    assert store.count() >= 20

    retriever = RAGRetriever(vector_store=store)

    # Benchmark queries covering each domain
    domain_queries = {
        "BILLING": "Where can I find and download my latest invoice receipt?",
        "SHIPPING": "How can I track the carrier delivering my package?",
        "CANCELLATION": "I need to cancel my order immediately before dispatch.",
        "ORDER": "How do I place an order online on the website?",
        "ACCOUNT": "I forgot my password and need to reset my login account.",
        "REFUND": "My package arrived broken and damaged. How do I get a refund?",
    }

    for expected_category, query_text in domain_queries.items():
        results = retriever.retrieve(query_text, top_k=3, min_score=0.10)
        assert len(results) > 0, f"No chunks retrieved for category {expected_category}"
        top_hit = results[0]
        assert (
            top_hit["category"] == expected_category
        ), f"Expected top-1 category {expected_category}, got {top_hit['category']} for query: '{query_text}'"
        assert top_hit["relevance_score"] > 0.30


def test_real_chroma_reset_operation(tmp_path):
    """Verify reset() empties collection and allows re-populating."""
    persist_dir = str(tmp_path / "reset_test")
    store = ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name="reset_kb",
    )
    store.add_documents([
        {
            "chunk_id": "test-chunk-1",
            "source_id": "test-doc",
            "title": "Temp",
            "category": "ACCOUNT",
            "content": "Temporary account content.",
            "metadata": {"intent": "edit_account"},
        }
    ])
    assert store.count() == 1

    store.reset()
    assert store.count() == 0


def test_retriever_low_score_filtering(real_ephemeral_store):
    """Verify retriever drops chunks below min_score and returns fallback context."""
    retriever = RAGRetriever(vector_store=real_ephemeral_store)
    # Query with impossible threshold
    chunks = retriever.retrieve("completely unrelated query about astronauts in orbit", min_score=0.99)
    assert chunks == []
    assert retriever.format_context_for_llm(chunks) == NO_RELEVANT_CONTEXT

