"""CLI script to run evaluation suite across golden test set."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Run RAG & Tool evaluation.")
    parser.add_argument("--test-set", default="data/eval/golden_test_set.json", help="Path to test set")
    args = parser.parse_args()

    print(f"Evaluation CLI invoked with test_set={args.test_set}")
    print("Evaluation pipeline will be implemented in Phase 6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
