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
    python ingestion_pipeline/chunk_syllabi.py --chunk-size 300 --overlap 50 --crns-file tamu_data/evals/eval_corpus.json --all
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion_pipeline.boilerplate_stripper import annotated_to_clean_markdown  # noqa: E402
from ingestion_pipeline.chunker_v3 import chunk_text, _tokens_approx  # noqa: E402

STEP2_ROOT = Path("tamu_data/processed/v3_step2_boilerplate")
OUTPUT_BASE = Path("tamu_data/processed")

# Filename pattern: {semester}_{dept}_{num}_{section}_{crn}_vNNN.md
_STEM_RE = re.compile(r"^(\d{6})_([A-Z]+_\d+)_\d+_(\d+)_")


def _parse_stem(stem: str) -> tuple[str, str, str]:
    """Extract (semester, course_id, crn) from a stem like 202611_CSCE_670_600_46627_v010."""
    m = _STEM_RE.match(stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return "", "", ""


def chunk_file(src: Path, chunk_size: int, overlap: int) -> dict:
    text = src.read_text(encoding="utf-8")
    text = annotated_to_clean_markdown(text)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    # Add token_count to each chunk
    for c in chunks:
        c["token_count"] = _tokens_approx(c["content"])

    clean_stem = src.stem  # e.g. 202611_CSCE_670_600_46627_v010
    semester, course_id, crn = _parse_stem(clean_stem)

    return {
        "source_file": src.name,
        "course_id": course_id,
        "semester": semester,
        "crn": crn,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }


_LOG_FIELDS = [
    "stem", "course_id", "semester", "crn", "source_file",
    "chunk_size_setting", "overlap_setting", "input_tokens",
    "chunk_count", "avg_chunk_tokens", "min_chunk_tokens", "max_chunk_tokens",
    "status", "error", "processed_at",
]


def main():
    parser = argparse.ArgumentParser(description="Re-chunk v3 stripped markdown files.")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--crns-file", metavar="PATH",
                        help="JSON file with a 'crns' array; restrict processing to those CRNs.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--files", type=int, default=5, metavar="N",
                       help="Process first N files (alphabetical). Default: 5.")
    group.add_argument("--all", action="store_true", help="Process all files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    # Load CRN filter if provided
    crn_filter: set[str] | None = None
    if args.crns_file:
        corpus = json.loads(Path(args.crns_file).read_text(encoding="utf-8"))
        crn_filter = set(corpus["crns"])

    # Collect source files: versioned .md files, excluding _stripped.md (alphabetical)
    sources = sorted(f for f in STEP2_ROOT.glob("*.md") if not f.name.endswith("_stripped.md"))
    if not sources:
        print(f"No versioned .md files found in {STEP2_ROOT}", file=sys.stderr)
        sys.exit(1)

    # Filter by CRN if requested
    if crn_filter is not None:
        sources = [f for f in sources if _parse_stem(f.stem)[2] in crn_filter]
        if not sources:
            print(f"No source files matched the CRNs in {args.crns_file}", file=sys.stderr)
            sys.exit(1)

    if not args.all:
        # Sample evenly across the full list so test runs cover diverse courses
        step = max(1, len(sources) // args.files)
        sources = sources[::step][: args.files]

    # Output folder
    date_str = datetime.now().strftime("%Y%m%d")
    out_dir = OUTPUT_BASE / f"v3_result_{args.chunk_size}t_{args.overlap}o_{date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_rows: list[dict] = []
    processed = skipped = 0
    for src in sources:
        clean_stem = src.stem
        out_path = out_dir / f"{clean_stem}.json"

        if out_path.exists() and not args.force:
            skipped += 1
            continue

        ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        row: dict = {
            "stem": clean_stem,
            "chunk_size_setting": args.chunk_size,
            "overlap_setting": args.overlap,
            "processed_at": ts,
        }
        try:
            raw_text = src.read_text(encoding="utf-8")
            row["input_tokens"] = _tokens_approx(annotated_to_clean_markdown(raw_text))
            result = chunk_file(src, args.chunk_size, args.overlap)
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

            token_counts = [c["token_count"] for c in result["chunks"]]
            row.update({
                "course_id": result["course_id"],
                "semester": result["semester"],
                "crn": result["crn"],
                "source_file": result["source_file"],
                "chunk_count": result["total_chunks"],
                "avg_chunk_tokens": round(sum(token_counts) / len(token_counts)) if token_counts else 0,
                "min_chunk_tokens": min(token_counts) if token_counts else 0,
                "max_chunk_tokens": max(token_counts) if token_counts else 0,
                "status": "ok",
                "error": "",
            })
            if token_counts:
                print(f"{clean_stem}: {result['total_chunks']} chunks, "
                      f"avg {row['avg_chunk_tokens']} tok "
                      f"(min {row['min_chunk_tokens']}, max {row['max_chunk_tokens']})")
            else:
                print(f"{clean_stem}: 0 chunks")
            processed += 1
        except Exception as exc:
            semester, course_id, crn = _parse_stem(clean_stem)
            row.update({
                "course_id": course_id,
                "semester": semester,
                "crn": crn,
                "source_file": src.name,
                "chunk_count": 0,
                "avg_chunk_tokens": 0,
                "min_chunk_tokens": 0,
                "max_chunk_tokens": 0,
                "status": "error",
                "error": str(exc),
            })
            print(f"ERROR {clean_stem}: {exc}", file=sys.stderr)
            processed += 1

        log_rows.append(row)

    # Write CSV log
    if log_rows:
        log_path = out_dir / "_chunk_log.csv"
        write_header = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_LOG_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(log_rows)
        print(f"Log → {log_path}")

    print(f"\nDone — {processed} processed, {skipped} skipped → {out_dir}")


if __name__ == "__main__":
    main()
