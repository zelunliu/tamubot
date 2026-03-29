"""Ingest multiple courses from boilerplate-stripped markdown files into a shared LightRAG storage.

Usage:
    python tools/lightrag_spike/ingest_multi.py --storage-dir storage_iter8_multi
    python tools/lightrag_spike/ingest_multi.py --storage-dir storage_iter8_multi --no-normalize

Options:
    --storage-dir   Storage directory name under tools/lightrag_spike/ (required)
    --gleaning      Entity extraction gleaning rounds (default: 3)
    --no-normalize  Skip post-processing entity normalization
"""

import argparse
import asyncio
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: F401
from wrappers import SYLLABUS_ENTITY_TYPES, amake_lightrag_improved, normalize_entity_names

SPIKE_DIR = Path(__file__).resolve().parent
BOILERPLATE_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "tamu_data/processed/v3_step2_boilerplate"
)

COURSE_CONFIGS = {
    "CSCE 670": (
        "202611_CSCE_670_600_46627_v010.md",
        "# CSCE 670 — Information Storage and Retrieval (Spring 2026)\n"
        "# This document is the official course syllabus for CSCE 670.\n\n",
    ),
    "CSCE 605": (
        "202541_CSCE_605_600_62077_v010.md",
        "# CSCE 605 — Compiler Design (Fall 2025)\n"
        "# This document is the official course syllabus for CSCE 605.\n\n",
    ),
    "CSCE 638": (
        "202611_CSCE_638_600_54988_v010.md",
        "# CSCE 638 — Natural Language Processing: Foundations and Techniques (Spring 2026)\n"
        "# This document is the official course syllabus for CSCE 638.\n\n",
    ),
}

DEFAULT_COURSES = ["CSCE 670", "CSCE 605", "CSCE 638"]


def load_course(course_id: str) -> str:
    filename, header = COURSE_CONFIGS[course_id]
    source = BOILERPLATE_DIR / filename
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    return header + source.read_text(encoding="utf-8")


async def run_ingestion(
    storage_dir: Path,
    course_ids: list[str],
    gleaning: int,
    run_normalize: bool,
) -> None:
    if storage_dir.exists():
        graph_file = storage_dir / "graph_chunk_entity_relation.graphml"
        if graph_file.exists():
            print(f"WARNING: {storage_dir} already has a graph. Deleting and re-ingesting.")
            shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    rag = await amake_lightrag_improved(
        working_dir=storage_dir,
        entity_types=SYLLABUS_ENTITY_TYPES,
        chunk_token_size=500,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=gleaning,
        patch_prompts=True,
        top_k=15,
        related_chunk_number=2,
    )

    total_start = time.time()
    for course_id in course_ids:
        print(f"\nIngesting {course_id}...")
        text = load_course(course_id)
        print(f"  Document: {len(text)} chars (~{len(text) // 4} tokens)")
        t0 = time.time()
        await rag.ainsert(text)
        print(f"  {course_id} done in {time.time() - t0:.1f}s")

    print(f"\nAll {len(course_ids)} courses ingested in {time.time() - total_start:.1f}s")

    graph_file = storage_dir / "graph_chunk_entity_relation.graphml"
    if graph_file.exists():
        print(f"Graph: {graph_file.stat().st_size / 1024:.1f} KB")

    if run_normalize:
        print("\nRunning entity name normalization...")
        result = normalize_entity_names(storage_dir)
        if result["merged"] > 0:
            print(f"  Merged {result['merged']} duplicate entities: {result['removed']}")
        else:
            print("  No case-drift duplicates found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest multiple courses into shared LightRAG storage")
    parser.add_argument("--storage-dir", required=True)
    parser.add_argument("--gleaning", type=int, default=3)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_ingestion(
        storage_dir=SPIKE_DIR / args.storage_dir,
        course_ids=DEFAULT_COURSES,
        gleaning=args.gleaning,
        run_normalize=not args.no_normalize,
    ))


if __name__ == "__main__":
    main()
