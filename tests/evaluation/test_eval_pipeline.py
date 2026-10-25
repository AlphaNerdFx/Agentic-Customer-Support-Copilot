"""Automated evaluation test suite verifying all PRD SLA targets on golden dataset."""

from __future__ import annotations

import chromadb
import pytest

from src.evaluation.dataset import load_golden_dataset
from src.evaluation.evaluator import CopilotEvaluator
from src.evaluation.report import generate_evaluation_report
from src.rag.chunking import chunk_article, deduplicate_and_enrich
from src.rag.ingest import DEFAULT_BITEXT_SEED_ROWS
from src.rag.retriever import RAGRetriever
from src.rag.store import ChromaVectorStore, DeterministicDomainEmbeddingFunction


@pytest.fixture(scope="module")
def seeded_eval_retriever():
    """Build a real ephemeral ChromaVectorStore seeded with canonical knowledge base."""
    client = chromadb.EphemeralClient()
    embed_fn = DeterministicDomainEmbeddingFunction(dim=384)
    store = ChromaVectorStore(
        collection_name="eval_test_kb",
        client=client,
        embedding_function=embed_fn,
    )
    articles = deduplicate_and_enrich(DEFAULT_BITEXT_SEED_ROWS)
    chunks = []
    for art in articles:
        chunks.extend(chunk_article(art))
    store.add_documents(chunks)
    return RAGRetriever(vector_store=store)


def test_load_golden_dataset_structure():
    """Verify golden evaluation dataset loads >= 50 scenarios with valid schemas."""
    cases = load_golden_dataset("data/eval/golden_test_set.json")
    assert len(cases) >= 50

    categories = {c.expected_category for c in cases}
    assert "ACCOUNT" in categories
    assert "BILLING" in categories
    assert "CANCELLATION" in categories
    assert "ORDER" in categories
    assert "REFUND" in categories
    assert "SHIPPING" in categories
    assert "OUT_OF_SCOPE" in categories
    assert "ADVERSARIAL" in categories

    actionable = [c for c in cases if c.expected_tool is not None]
    assert len(actionable) >= 15
    tools = {c.expected_tool for c in actionable}
    assert "initiate_refund" in tools
    assert "escalate_ticket" in tools
    assert "send_followup_email" in tools


def test_copilot_evaluator_meets_all_prd_slas(seeded_eval_retriever, tmp_path):
    """Run full evaluation suite and assert that every PRD target SLA is satisfied."""
    test_cases = load_golden_dataset("data/eval/golden_test_set.json")
    evaluator = CopilotEvaluator(retriever=seeded_eval_retriever)

    eval_data = evaluator.evaluate_all(test_cases)
    summary = eval_data["summary"]

    # 1. Citation Accuracy (Target >= 85%)
    assert (
        summary["citation_accuracy"] >= 0.85
    ), f"Citation accuracy {summary['citation_accuracy']} fell below 0.85"

    # 2. Task Completion Rate (Target >= 70%)
    assert (
        summary["task_completion_rate"] >= 0.70
    ), f"Task completion rate {summary['task_completion_rate']} fell below 0.70"

    # 3. HITL Gate Coverage (Target == 100%)
    assert (
        summary["hitl_gate_coverage"] == 1.00
    ), f"HITL gate coverage {summary['hitl_gate_coverage']} must be exactly 1.00"

    # 4. False-Positive Gate Rate (Target <= 5%)
    assert (
        summary["false_positive_interrupt_rate"] <= 0.05
    ), f"False-positive interrupt rate {summary['false_positive_interrupt_rate']} exceeded 0.05"

    # 5. Average Quality Score (Target >= 3.5 / 5.0)
    assert (
        summary["avg_quality_score"] >= 3.5
    ), f"Quality score {summary['avg_quality_score']} fell below 3.5"

    # 6. Latency SLA (p90 < 3000 ms)
    assert (
        summary["latency_p90_ms"] < 3000.0
    ), f"Latency p90 {summary['latency_p90_ms']} ms exceeded 3000 ms"

    # 7. Overall SLA flag
    assert summary["sla_targets_met"] is True

    # 8. Verify report generation
    json_out = str(tmp_path / "test_report.json")
    md_out = str(tmp_path / "test_report.md")
    md_content = generate_evaluation_report(
        eval_data, json_path=json_out, md_path=md_out
    )
    assert "Executive SLA Summary" in md_content
    assert "CV Achievement Bullet Point Earned" in md_content


def test_deterministic_quality_score_rubric():
    """Verify deterministic rubric scoring under varied completeness and safety states."""
    from src.evaluation.dataset import GoldenTestCase
    from src.evaluation.evaluator import deterministic_quality_score

    # Refusal test case
    case_refusal = GoldenTestCase(
        id="test-refusal-1",
        query="Random recipe query",
        expected_category="OUT_OF_SCOPE",
        expected_source_ids=[],
        expected_tool=None,
        expect_refusal=True,
        contains_pii=False,
        required_keywords=["enough information"],
    )
    score_refusal = deterministic_quality_score(
        test_case=case_refusal,
        assistant_content="I don't have enough information in our knowledge base to answer this.",
        citations=[],
        tool_status="none",
        proposed_tool=None,
    )
    assert score_refusal == 5.0

    # Incomplete reply (missing citations and keywords)
    case_info = GoldenTestCase(
        id="test-info-1",
        query="How do I get a refund?",
        expected_category="REFUND",
        expected_source_ids=["kb-refund-01"],
        expected_tool=None,
        expect_refusal=False,
        contains_pii=False,
        required_keywords=["refund", "policy"],
    )
    score_bad = deterministic_quality_score(
        test_case=case_info,
        assistant_content="Hello world.",
        citations=[],
        tool_status="none",
        proposed_tool=None,
    )
    assert score_bad <= 3.0


def test_copilot_evaluator_single_case_evaluation(seeded_eval_retriever):
    """Verify single test case evaluation output schema and attributes."""
    from src.evaluation.dataset import GoldenTestCase

    evaluator = CopilotEvaluator(retriever=seeded_eval_retriever)
    case = GoldenTestCase(
        id="test-single-1",
        query="What is your return policy for damaged items?",
        expected_category="REFUND",
        expected_source_ids=["kb-refund-refund_damaged_item"],
        expected_tool=None,
        expect_refusal=False,
        contains_pii=False,
        required_keywords=["refund"],
    )

    result = evaluator.evaluate_test_case(case)
    assert result["id"] == "test-single-1"
    assert result["category"] == "REFUND"
    assert result["citation_correct"] is True
    assert result["task_completed"] is True
    assert result["gate_correct"] is True
    assert result["quality_score"] >= 4.0
    assert result["latency_ms"] >= 0.0
