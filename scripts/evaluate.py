"""CLI script to run full evaluation suite across golden test set."""

from __future__ import annotations

import argparse
import sys

from src.evaluation.dataset import load_golden_dataset
from src.evaluation.evaluator import CopilotEvaluator
from src.evaluation.report import generate_evaluation_report


def main() -> int:
    """Run golden test set evaluation, generate reports, and verify SLA targets."""
    parser = argparse.ArgumentParser(description="Run RAG & Tool evaluation.")
    parser.add_argument(
        "--test-set",
        default="data/eval/golden_test_set.json",
        help="Path to golden test set JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of test cases to run (e.g. --limit 5 for <$0.01 live API testing)",
    )
    args = parser.parse_args()

    print(f"Loading golden evaluation test cases from: {args.test_set}")
    test_cases = load_golden_dataset(args.test_set)
    if args.limit and args.limit > 0:
        test_cases = test_cases[: args.limit]
        print(f"Limiting evaluation run to first {len(test_cases)} scenarios (--limit {args.limit}).")
    else:
        print(f"Loaded {len(test_cases)} annotated scenarios.")

    evaluator = CopilotEvaluator()
    print("Executing evaluation across real ChromaDB + LangGraph infrastructure...")
    eval_data = evaluator.evaluate_all(test_cases)

    generate_evaluation_report(eval_data)
    summary = eval_data["summary"]

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Scenarios:         {summary['total_test_cases']}")
    print(f"Citation Accuracy:       {summary['citation_accuracy'] * 100:.1f}% (Target: >=85.0%)")
    print(f"Task Completion Rate:    {summary['task_completion_rate'] * 100:.1f}% (Target: >=70.0%)")
    print(f"HITL Gate Coverage:      {summary['hitl_gate_coverage'] * 100:.1f}% (Target: 100.0%)")
    print(f"False-Positive Gates:    {summary['false_positive_interrupt_rate'] * 100:.1f}% (Target: <=5.0%)")
    print(f"Average Quality Score:   {summary['avg_quality_score']:.2f} / 5.0 (Target: >=3.5)")
    print(f"Turn Latency (p90):      {summary['latency_p90_ms']:.1f} ms (Target: <3000 ms)")
    print(f"All SLA Targets Passed:  {summary['sla_targets_met']}")
    print("=" * 60)

    print("\nReports written to:")
    print("  - data/eval/evaluation_report.json")
    print("  - data/eval/evaluation_report.md")

    return 0 if summary["sla_targets_met"] else 1


if __name__ == "__main__":
    sys.exit(main())
