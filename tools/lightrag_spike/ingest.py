"""Ingest all 19 eval-set CSCE courses into a shared LightRAG storage.

Uses boilerplate-stripped .md files from tamu_data/processed/v3_step2_boilerplate/.
CSCE 650 falls back to MongoDB chunks_v3 (no boilerplate file available).

Usage:
    python tools/lightrag_spike/ingest.py [--storage-dir storage_19course] [--dry-run]

Options:
    --storage-dir   Storage dir name under tools/lightrag_spike/ (default: storage_19course)
    --dry-run       Print what would be ingested without making any API calls
    --gleaning      Entity extraction gleaning rounds (default: 1, minimum cost)
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: F401
from wrappers import SYLLABUS_ENTITY_TYPES, amake_lightrag_improved

SPIKE_DIR = Path(__file__).resolve().parent
BOILERPLATE_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "tamu_data/processed/v3_step2_boilerplate"
)

# 18 courses with boilerplate .md files (Spring 2026 section 600)
BOILERPLATE_COURSES = {
    "CSCE 608": "202611_CSCE_608_600_46648_v010.md",
    "CSCE 611": "202611_CSCE_611_600_50668_v010.md",
    "CSCE 612": "202611_CSCE_612_600_42640_v010.md",
    "CSCE 624": "202611_CSCE_624_600_58435_v010.md",
    "CSCE 629": "202611_CSCE_629_600_54978_v010.md",
    "CSCE 632": "202611_CSCE_632_600_54784_v010.md",
    "CSCE 633": "202611_CSCE_633_601_58706_v010.md",
    "CSCE 638": "202611_CSCE_638_600_54988_v010.md",
    "CSCE 641": "202611_CSCE_641_600_58437_v010.md",
    "CSCE 656": "202611_CSCE_656_600_42432_v010.md",
    "CSCE 665": "202611_CSCE_665_600_30874_v010.md",
    "CSCE 669": "202611_CSCE_669_600_58438_v010.md",
    "CSCE 670": "202611_CSCE_670_600_46627_v010.md",
    "CSCE 672": "202611_CSCE_672_600_58439_v010.md",
    "CSCE 676": "202611_CSCE_676_600_58440_v010.md",
    "CSCE 679": "202611_CSCE_679_600_55144_v010.md",
    "CSCE 681": "202611_CSCE_681_600_12644_v010.md",
    "CSCE 713": "202611_CSCE_713_600_58633_v010.md",
}

# CSCE 650: no boilerplate file, fetched from MongoDB
MONGO_COURSES = ["CSCE 650"]


def load_from_boilerplate(course_id: str) -> str:
    filename = BOILERPLATE_COURSES[course_id]
    path = BOILERPLATE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Boilerplate file not found: {path}")
    header = f"# {course_id} — Course Syllabus (Spring 2026)\n# This document is the official course syllabus for {course_id}.\n\n"
    return header + path.read_text(encoding="utf-8")


def load_from_mongo(course_id: str) -> str:
    import pymongo
    client = pymongo.MongoClient(config.MONGODB_URI)
    db = client[config.MONGODB_DB]
    chunks = list(
        db["chunks_v3"]
        .find({"course_id": course_id}, {"_id": 0, "anchor": 1, "content": 1, "chunk_index": 1})
        .sort("chunk_index", 1)
    )
    client.close()
    if not chunks:
        raise ValueError(f"No chunks found in MongoDB for {course_id}")
    header = f"# {course_id} — Course Syllabus\n# This document is the official course syllabus for {course_id}.\n\n"
    body = "\n\n".join(f"{c.get('anchor', '')}\n\n{c.get('content', '')}" for c in chunks)
    return header + body


def load_course(course_id: str) -> tuple[str, str]:
    """Returns (text, source_label)."""
    if course_id in BOILERPLATE_COURSES:
        return load_from_boilerplate(course_id), "boilerplate"
    return load_from_mongo(course_id), "mongodb"


async def run_ingestion(storage_dir: Path, gleaning: int, dry_run: bool) -> None:
    all_courses = list(BOILERPLATE_COURSES.keys()) + MONGO_COURSES

    if dry_run:
        print(f"[DRY RUN] Would ingest {len(all_courses)} courses into {storage_dir}")
        for cid in all_courses:
            src = "boilerplate" if cid in BOILERPLATE_COURSES else "mongodb"
            print(f"  {cid} ({src})")
        return

    if storage_dir.exists():
        graph_file = storage_dir / "graph_chunk_entity_relation.graphml"
        if graph_file.exists():
            print(f"WARNING: {storage_dir} already has a graph. Delete it to re-ingest.")
            print("Aborting — use a fresh storage dir or delete the existing one.")
            return

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
    for i, course_id in enumerate(all_courses, 1):
        print(f"[{i}/{len(all_courses)}] Ingesting {course_id}...")
        try:
            text, src = load_course(course_id)
            print(f"  Source: {src}, {len(text)} chars (~{len(text) // 4} tokens)")
            t0 = time.time()
            await rag.ainsert(text)
            print(f"  Done in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")

    elapsed = time.time() - total_start
    print(f"\nIngestion complete in {elapsed:.1f}s")
    graph_file = storage_dir / "graph_chunk_entity_relation.graphml"
    if graph_file.exists():
        print(f"Graph: {graph_file.stat().st_size / 1024:.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest 19 eval-set courses into LightRAG")
    parser.add_argument("--storage-dir", default="storage_19course",
                        help="Storage dir name under tools/lightrag_spike/ (default: storage_19course)")
    parser.add_argument("--gleaning", type=int, default=1,
                        help="Gleaning rounds (default: 1 for minimum cost)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_ingestion(
        storage_dir=SPIKE_DIR / args.storage_dir,
        gleaning=args.gleaning,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
