"""Ingest CSCE 670 from the boilerplate-stripped markdown file into a LightRAG storage dir.

Usage:
    python tools/lightrag_spike/ingest_670.py --storage-dir storage_iter1 [options]

Options:
    --storage-dir   Storage directory name under tools/lightrag_spike/ (required)
    --chunk-size    LightRAG internal chunk token size (default: 2000)
    --overlap       Chunk overlap tokens (default: 200)
    --gleaning      Entity extraction gleaning rounds (default: 1)
    --entity-types  Comma-separated entity types (default: SYLLABUS_ENTITY_TYPES)
    --no-patch      Skip prompt patching (use LightRAG defaults)
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: F401 — ensures env is loaded
from wrappers import SYLLABUS_ENTITY_TYPES, amake_lightrag_improved

# Boilerplate-stripped source document for CSCE 670
SOURCE_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "tamu_data/processed/v3_step2_boilerplate/202611_CSCE_670_600_46627_v010.md"
)
SPIKE_DIR = Path(__file__).resolve().parent


COURSE_CONTEXT_HEADER = (
    "# CSCE 670 — Information Storage and Retrieval (Spring 2026)\n"
    "# This document is the official course syllabus for CSCE 670.\n\n"
)


def load_source() -> str:
    """Load the CSCE 670 boilerplate-stripped markdown document.

    Prepends a course context header so every internal chunk produced by
    LightRAG's tokenizer contains the course identifier, preventing
    cross-course entity mixing when the document is split.
    """
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")
    raw = SOURCE_FILE.read_text(encoding="utf-8")
    return COURSE_CONTEXT_HEADER + raw


async def run_ingestion(
    storage_dir: Path,
    chunk_token_size: int,
    chunk_overlap_token_size: int,
    entity_extract_max_gleaning: int,
    entity_types: list[str],
    patch_prompts: bool,
    top_k: int = 40,
    related_chunk_number: int = 5,
) -> None:
    text = load_source()
    print(f"Source: {SOURCE_FILE.name}")
    print(f"Document length: {len(text)} chars (~{len(text)//4} tokens)")
    print(f"Storage: {storage_dir}")
    print(f"Config: chunk={chunk_token_size}, overlap={chunk_overlap_token_size}, "
          f"gleaning={entity_extract_max_gleaning}, patch_prompts={patch_prompts}")
    print(f"Entity types: {entity_types}")
    print()

    if storage_dir.exists():
        graph_file = storage_dir / "graph_chunk_entity_relation.graphml"
        if graph_file.exists():
            print(f"WARNING: {storage_dir} already has a graph. Deleting and re-ingesting.")
            import shutil
            shutil.rmtree(storage_dir)

    storage_dir.mkdir(parents=True, exist_ok=True)

    rag = await amake_lightrag_improved(
        working_dir=storage_dir,
        entity_types=entity_types,
        chunk_token_size=chunk_token_size,
        chunk_overlap_token_size=chunk_overlap_token_size,
        entity_extract_max_gleaning=entity_extract_max_gleaning,
        patch_prompts=patch_prompts,
        top_k=top_k,
        related_chunk_number=related_chunk_number,
    )

    print("Ingesting CSCE 670...")
    t0 = time.time()
    await rag.ainsert(text)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    # Quick stats
    graph_file = storage_dir / "graph_chunk_entity_relation.graphml"
    if graph_file.exists():
        size_kb = graph_file.stat().st_size / 1024
        print(f"Graph file: {size_kb:.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CSCE 670 into LightRAG (iteration mode)")
    parser.add_argument("--storage-dir", required=True, help="Storage dir name under tools/lightrag_spike/")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--gleaning", type=int, default=1)
    parser.add_argument("--entity-types", type=str, default=None,
                        help="Comma-separated types (default: SYLLABUS_ENTITY_TYPES)")
    parser.add_argument("--no-patch", action="store_true", help="Skip prompt patching")
    parser.add_argument("--top-k", type=int, default=40, help="top_k for retrieval (default: 40)")
    parser.add_argument("--related-chunks", type=int, default=5, help="Raw text chunks to include in context (default: 5)")
    args = parser.parse_args()

    storage_dir = SPIKE_DIR / args.storage_dir
    entity_types = (
        [t.strip() for t in args.entity_types.split(",")]
        if args.entity_types
        else SYLLABUS_ENTITY_TYPES
    )

    asyncio.run(run_ingestion(
        storage_dir=storage_dir,
        chunk_token_size=args.chunk_size,
        chunk_overlap_token_size=args.overlap,
        entity_extract_max_gleaning=args.gleaning,
        entity_types=entity_types,
        patch_prompts=not args.no_patch,
        top_k=args.top_k,
        related_chunk_number=args.related_chunks,
    ))


if __name__ == "__main__":
    main()
