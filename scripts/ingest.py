"""CLI script to ingest Bitext customer support dataset into ChromaDB."""

from __future__ import annotations

import argparse
import sys

from src.core.config import settings
from src.rag.ingest import run_ingestion


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest customer support knowledge base into ChromaDB."
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=settings.chroma_persist_dir,
        help="Directory to persist ChromaDB collection",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="data/knowledge_base/articles.json",
        help="Path to export deduplicated articles JSON",
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        default=False,
        help="Attempt loading full Bitext dataset from HuggingFace",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=True,
        help="Enable query-enriched deduplication",
    )

    args = parser.parse_args()

    print("=" * 65)
    print(" FDE Copilot Knowledge Base Ingestion Pipeline (Milestone 1)")
    print("=" * 65)
    print(f"Persist Directory:   {args.persist_dir}")
    print(f"Output JSON Path:    {args.output_json}")
    print(f"HuggingFace Source:  {args.use_hf}")
    print(f"Deduplication:       {args.deduplicate}")
    print("-" * 65)

    try:
        stats = run_ingestion(
            persist_dir=args.persist_dir,
            output_json_path=args.output_json,
            use_hf=args.use_hf,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 65)
    print(" Ingestion Summary")
    print("=" * 65)
    print(f" Raw Rows Processed:       {stats['raw_rows']}")
    print(f" Filtered Operational Rows: {stats['filtered_rows']}")
    print(f" Consolidated Articles:    {stats['articles_created']}")
    print(f" Chunks Indexed in Chroma: {stats['chunks_indexed']}")
    print(f" Operational Categories:   {', '.join(stats['categories'])}")
    print(f" Total Embedding Tokens:   {stats['total_embedding_tokens']}")
    print(f" Pipeline Latency:         {stats['elapsed_ms']} ms")
    print("=" * 65)
    print("[SUCCESS] Knowledge base ingestion completed successfully.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
