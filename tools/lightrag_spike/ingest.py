"""Ingest eval-set CSCE courses from MongoDB chunks_v3 into LightRAG storage.

Usage:
    python tools/lightrag_spike/ingest.py [--dry-run]

Fetches chunks for the 19 eval-set CSCE courses and inserts them into LightRAG.
Runs entity extraction + relationship building via TAMU gateway LLM.
Graph + vectors are persisted to tools/lightrag_spike/storage/.

Cost note: ~380 LLM calls (one per chunk) via TAMU gateway. One-time cost.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import pymongo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config

# Add spike dir to path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wrappers import WORKING_DIR, make_lightrag

# The 19 courses from the eval golden set
EVAL_COURSE_IDS = [
    "CSCE 608",
    "CSCE 611",
    "CSCE 612",
    "CSCE 624",
    "CSCE 629",
    "CSCE 632",
    "CSCE 633",
    "CSCE 638",
    "CSCE 641",
    "CSCE 650",
    "CSCE 656",
    "CSCE 665",
    "CSCE 669",
    "CSCE 670",
    "CSCE 672",
    "CSCE 676",
    "CSCE 679",
    "CSCE 681",
    "CSCE 713",
]


def fetch_chunks(course_ids: list[str]) -> list[dict]:
    """Fetch all chunks for the given course IDs from MongoDB chunks_v3."""
    client = pymongo.MongoClient(config.MONGODB_URI)
    db = client[config.MONGODB_DB]
    collection = db["chunks_v3"]
    chunks = list(collection.find(
        {"course_id": {"$in": course_ids}},
        {"_id": 0, "anchor": 1, "content": 1, "course_id": 1,
         "section": 1, "term": 1, "category": 1, "chunk_index": 1},
    ).sort([("course_id", 1), ("chunk_index", 1)]))
    client.close()
    return chunks


def format_chunk(chunk: dict) -> str:
    """Format a chunk as LightRAG input text."""
    anchor = chunk.get("anchor", "")
    content = chunk.get("content", "")
    return f"{anchor}\n\n{content}"


async def run_ingestion(dry_run: bool = False) -> None:
    chunks = fetch_chunks(EVAL_COURSE_IDS)
    print(f"Fetched {len(chunks)} chunks from MongoDB across {len(EVAL_COURSE_IDS)} courses")

    if dry_run:
        print("[DRY RUN] Skipping LightRAG insertion.")
        for cid in EVAL_COURSE_IDS:
            n = sum(1 for c in chunks if c["course_id"] == cid)
            print(f"  {cid}: {n} chunks")
        return

    # Check for existing storage to avoid re-ingesting
    graph_file = WORKING_DIR / "graph_chunk_entity_relation.graphml"
    if graph_file.exists():
        print(f"WARNING: {graph_file} already exists.")
        print("Delete tools/lightrag_spike/storage/ to re-ingest from scratch.")
        answer = input("Continue and add to existing graph? [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    rag = make_lightrag()
    print(f"LightRAG storage: {WORKING_DIR}")

    # Group by course for progress logging
    by_course: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_course.setdefault(chunk["course_id"], []).append(chunk)

    total_start = time.time()
    for i, (course_id, course_chunks) in enumerate(sorted(by_course.items()), 1):
        print(f"[{i}/{len(by_course)}] Ingesting {course_id} ({len(course_chunks)} chunks)...")
        course_start = time.time()
        for chunk in course_chunks:
            text = format_chunk(chunk)
            await rag.ainsert(text)
        elapsed = time.time() - course_start
        print(f"  Done in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\nIngestion complete: {len(chunks)} chunks in {total_elapsed:.1f}s")
    print(f"Storage written to: {WORKING_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest eval-set CSCE courses into LightRAG")
    parser.add_argument("--dry-run", action="store_true", help="Fetch chunks but skip LightRAG insertion")
    args = parser.parse_args()
    asyncio.run(run_ingestion(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
