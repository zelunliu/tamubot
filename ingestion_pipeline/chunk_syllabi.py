"""
ingestion_pipeline/chunk_syllabi.py

Standalone re-chunker for v3 syllabus pipeline.

Reads clean syllabus *_vNNN.md files from v3_step2_boilerplate/ (not _stripped.md),
chunks them with configurable size/overlap, and writes JSON to a date-stamped output folder.

No LLM calls. Output is for experimentation and inspection only.

Usage:
    python ingestion_pipeline/chunk_syllabi.py --chunk-size 300 --overlap 50
    python ingestion_pipeline/chunk_syllabi.py --chunk-size 300 --overlap 50 --files 10
    python ingestion_pipeline/chunk_syllabi.py --chunk-size 300 --overlap 50 --all
    python ingestion_pipeline/chunk_syllabi.py --chunk-size 500 --overlap 75 --all --force
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion_pipeline.boilerplate_stripper import annotated_to_clean_markdown  # noqa: E402
from ingestion_pipeline.chunker_v3 import chunk_text, _tokens_approx  # noqa: E402

STEP2_ROOT = Path("tamu_data/processed/v3_step2_boilerplate")
OUTPUT_BASE = Path("tamu_data/processed")

# Filename pattern: {semester}_{dept}_{num}_{section}_{crn}_vNNN_stripped.md
_STEM_RE = re.compile(r"^(\d{6})_([A-Z]+_\d+)_")


def _parse_stem(stem: str) -> tuple[str, str]:
    """Extract (semester, course_id) from a stripped stem like 202611_CSCE_670_600_46627_v010."""
    m = _STEM_RE.match(stem)
    if m:
        return m.group(1), m.group(2)
    return "", ""


def chunk_file(src: Path, chunk_size: int, overlap: int) -> dict:
    text = src.read_text(encoding="utf-8")
    text = annotated_to_clean_markdown(text)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    # Add token_count to each chunk
    for c in chunks:
        c["token_count"] = _tokens_approx(c["content"])

    # Stem without _stripped suffix
    clean_stem = src.stem  # e.g. 202611_CSCE_670_600_46627_v010
    semester, course_id = _parse_stem(clean_stem)

    return {
        "source_file": src.name,  # e.g. 202611_CSCE_670_600_46627_v010.md
        "course_id": course_id,
        "semester": semester,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }


def main():
    parser = argparse.ArgumentParser(description="Re-chunk v3 stripped markdown files.")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--files", type=int, default=5, metavar="N",
                       help="Process first N files (alphabetical). Default: 5.")
    group.add_argument("--all", action="store_true", help="Process all files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    # Collect source files: versioned .md files, excluding _stripped.md (alphabetical)
    sources = sorted(f for f in STEP2_ROOT.glob("*.md") if not f.name.endswith("_stripped.md"))
    if not sources:
        print(f"No *_stripped.md files found in {STEP2_ROOT}", file=sys.stderr)
        sys.exit(1)

    if not args.all:
        # Sample evenly across the full list so test runs cover diverse courses
        step = max(1, len(sources) // args.files)
        sources = sources[::step][: args.files]

    # Output folder
    date_str = datetime.now().strftime("%Y%m%d")
    out_dir = OUTPUT_BASE / f"v3_result_{args.chunk_size}t_{args.overlap}o_{date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = skipped = 0
    for src in sources:
        clean_stem = src.stem.replace("_stripped", "")
        out_path = out_dir / f"{clean_stem}.json"

        if out_path.exists() and not args.force:
            skipped += 1
            continue

        result = chunk_file(src, args.chunk_size, args.overlap)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        token_counts = [c["token_count"] for c in result["chunks"]]
        if token_counts:
            avg = round(sum(token_counts) / len(token_counts))
            print(f"{clean_stem}: {result['total_chunks']} chunks, "
                  f"avg {avg} tok (min {min(token_counts)}, max {max(token_counts)})")
        else:
            print(f"{clean_stem}: 0 chunks")
        processed += 1

    print(f"\nDone — {processed} processed, {skipped} skipped → {out_dir}")


if __name__ == "__main__":
    main()
