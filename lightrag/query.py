"""Interactive LightRAG query CLI — runs query in all 4 modes side-by-side.

Usage:
    python tools/lightrag_spike/query.py "what are the grading policies for CSCE 670?"
    python tools/lightrag_spike/query.py "which courses cover indexing algorithms?" --mode hybrid

Requires: ingest.py must have been run first (storage/ must exist).
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lightrag import QueryParam

# Add spike dir to path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wrappers import WORKING_DIR, make_lightrag

MODES = ["naive", "local", "global", "hybrid"]


async def run_query(query: str, modes: list[str]) -> None:
    if not WORKING_DIR.exists():
        print(f"ERROR: storage not found at {WORKING_DIR}")
        print("Run ingest.py first.")
        sys.exit(1)

    rag = make_lightrag()

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*60}")
        result = await rag.aquery(query, param=QueryParam(mode=mode))
        print(result)

    print(f"\n{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query LightRAG in one or all modes")
    parser.add_argument("query", help="Query string")
    parser.add_argument(
        "--mode",
        choices=MODES + ["all"],
        default="all",
        help="Query mode (default: all)",
    )
    args = parser.parse_args()

    modes = MODES if args.mode == "all" else [args.mode]
    asyncio.run(run_query(args.query, modes))


if __name__ == "__main__":
    main()
