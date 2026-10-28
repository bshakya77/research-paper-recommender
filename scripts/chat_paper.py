"""Stage 6 CLI: ask follow-up questions about one paper in a LangChain session."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat import chat_with_paper
from src.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a paper-scoped follow-up chat")
    parser.add_argument("--paper-id", required=True, help="Canonical paper ID, such as an arXiv ID")
    parser.add_argument("--top-k", type=int, default=settings.top_k_retrieval,
                        help="Evidence chunks to retrieve per turn")
    args = parser.parse_args()

    session_id = None
    print("Ask follow-up questions about this paper. Enter /exit to finish.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"/exit", "/quit"}:
            break
        if not question:
            continue

        try:
            response = chat_with_paper(
                args.paper_id,
                question,
                session_id=session_id,
                top_k=args.top_k,
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            sys.exit(str(exc))

        session_id = response.session_id
        print(f"\nAssistant ({response.provider}/{response.model}):\n{response.answer}")
        print("\nSources:")
        for number, ref in enumerate(response.evidence_refs, start=1):
            print(f"[{number}] {ref.section}, page {ref.page}: {ref.snippet}")


if __name__ == "__main__":
    main()
