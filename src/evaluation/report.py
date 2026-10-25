"""Report generation formatting evaluation metrics into JSON and Markdown artifacts."""

from __future__ import annotations

import json
import os
from typing import Any


def format_markdown_report(eval_data: dict[str, Any]) -> str:
    """Format evaluation run results into a structured Markdown document."""
    summary = eval_data.get("summary", {})
    categories = eval_data.get("category_breakdown", {})

    total_cases = summary.get("total_test_cases", 0)
    task_comp_pct = f"{summary.get('task_completion_rate', 0.0) * 100:.1f}"
    citation_acc_pct = f"{summary.get('citation_accuracy', 0.0) * 100:.1f}"
    hitl_cov_pct = f"{summary.get('hitl_gate_coverage', 0.0) * 100:.1f}"
    fp_interrupt_pct = f"{summary.get('false_positive_interrupt_rate', 0.0) * 100:.1f}"
    quality_score = f"{summary.get('avg_quality_score', 0.0):.2f}"
    p50_ms = f"{summary.get('latency_p50_ms', 0.0):.1f}"
    p90_ms = f"{summary.get('latency_p90_ms', 0.0):.1f}"
    p99_ms = f"{summary.get('latency_p99_ms', 0.0):.1f}"
    avg_ms = f"{summary.get('avg_latency_ms', 0.0):.1f}"

    def badge(passed: bool) -> str:
        return "✅ **PASSED**" if passed else "❌ **FAILED**"

    sla = summary.get("sla_details", {})

    md_lines = [
        "# 🛡️ FDE Support Copilot: Golden Dataset Evaluation Report",
        "",
        "> **Evaluation Target Specification:** [PRD.md](../../docs/PRD.md) & [AGENTS.md](../../AGENTS.md)",
        "> **Infrastructure:** Zero-Mock ChromaDB Vector Store + LangGraph State Machine Checkpointer",
        "",
        "---",
        "",
        "## 1. Executive SLA Summary",
        "",
        "| Metric | SLA Target | Measured Value | Compliance Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Citation Accuracy** | $\\ge 85.0\\%$ | **{citation_acc_pct}%** | {badge(sla.get('citation_accuracy_met', False))} |",
        f"| **Task Completion Rate** | $\\ge 70.0\\%$ | **{task_comp_pct}%** | {badge(sla.get('task_completion_rate_met', False))} |",
        f"| **HITL Gate Coverage** | $100.0\\%$ | **{hitl_cov_pct}%** | {badge(sla.get('hitl_gate_coverage_met', False))} |",
        f"| **False-Positive Interrupts** | $\\le 5.0\\%$ | **{fp_interrupt_pct}%** | {badge(summary.get('false_positive_interrupt_rate', 1.0) <= 0.05)} |",
        f"| **Response Quality Score** | $\\ge 3.50 / 5.0$ | **{quality_score} / 5.0** | {badge(sla.get('avg_quality_score_met', False))} |",
        f"| **Latency (p90)** | $< 3000\\text{{ ms}}$ | **{p90_ms} ms** | {badge(sla.get('latency_p90_met', False))} |",
        "",
        "---",
        "",
        "## 2. Latency & Performance Breakdown",
        "",
        "| Metric | Measured Value |",
        "| :--- | :--- |",
        f"| **p50 Median Turn Latency** | `{p50_ms} ms` |",
        f"| **p90 Turn Latency** | `{p90_ms} ms` |",
        f"| **p99 Worst-Case Turn Latency** | `{p99_ms} ms` |",
        f"| **Mean Turn Latency** | `{avg_ms} ms` |",
        f"| **Evaluated Scenarios** | `{total_cases} test cases` |",
        "",
        "---",
        "",
        "## 3. Operational Category Breakdown",
        "",
        "| Category | Scenarios | Citation Accuracy | Task Completion | Avg Quality |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for cat_name, metrics in categories.items():
        c_count = metrics.get("count", 0)
        c_cit = f"{metrics.get('citation_accuracy', 0.0) * 100:.1f}%"
        c_comp = f"{metrics.get('task_completion_rate', 0.0) * 100:.1f}%"
        c_qual = f"{metrics.get('avg_quality_score', 0.0):.2f}"
        md_lines.append(
            f"| `{cat_name}` | {c_count} | {c_cit} | {c_comp} | {c_qual} |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 4. CV Achievement Bullet Point Earned (Per AGENTS.md)",
            "",
            "> *\"Designed and deployed an embedded support copilot with RAG over a 40K-article knowledge base, citation-grounded reply drafting, and human-in-the-loop approval for irreversible actions. Achieved **"
            + task_comp_pct
            + "%** task completion rate across **"
            + str(total_cases)
            + "** test tasks with **"
            + citation_acc_pct
            + "%** citation accuracy.\"*",
            "",
        ]
    )

    return "\n".join(md_lines)


def generate_evaluation_report(
    eval_data: dict[str, Any],
    json_path: str = "data/eval/evaluation_report.json",
    md_path: str = "data/eval/evaluation_report.md",
) -> str:
    """Serialize evaluation results into JSON and Markdown files on disk."""
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)

    md_content = format_markdown_report(eval_data)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return md_content


__all__ = ["format_markdown_report", "generate_evaluation_report"]
