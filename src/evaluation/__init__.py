"""Evaluation package for golden test dataset, benchmarking, and reporting."""

from src.evaluation.dataset import GoldenTestCase, load_golden_dataset
from src.evaluation.evaluator import CopilotEvaluator
from src.evaluation.report import (
    format_markdown_report,
    generate_evaluation_report,
)

__all__ = [
    "CopilotEvaluator",
    "GoldenTestCase",
    "format_markdown_report",
    "generate_evaluation_report",
    "load_golden_dataset",
]
