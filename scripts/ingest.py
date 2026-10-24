"""CLI script to ingest Bitext customer support dataset into ChromaDB."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Ingest knowledge base articles.")
    parser.add_argument("--deduplicate", action="store_true", default=True, help="Enable query-enriched deduplication")
    args = parser.parse_args()

    print(f"Ingestion CLI invoked with deduplicate={args.deduplicate}")
    print("Ingestion pipeline will be implemented in Phase 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
