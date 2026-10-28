"""Stage 3 CLI: build or incrementally update the FAISS index."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indexing import build_index
from src.vectorstore import get_vectorstore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="Optional smoke-test query after building")
    ap.add_argument("--rebuild", action="store_true",
                    help="Recreate the index from the full catalog, including already indexed papers")
    args = ap.parse_args()

    try:
        report = build_index(rebuild=args.rebuild)
    except ValueError as exc:
        sys.exit(str(exc))
    if report.added_chunks:
        print(
            f"Added {report.added_chunks} chunks for {report.added_papers} papers; "
            f"index now has {report.total_chunks} chunks across {report.total_papers} papers"
        )
    else:
        print(f"Index is up to date ({report.total_chunks} chunks)")

    if args.query:
        for r in get_vectorstore().search(args.query, top_k=5, chunk_type="abstract"):
            print(f"{r.score:.3f}  {r.chunk.paper_id}  {r.chunk.text.splitlines()[0][:90]}")


if __name__ == "__main__":
    main()
