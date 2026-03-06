"""
rebuild_csv.py — Rebuild parsing_progress.csv from all JSONs in today's output dir.

Usage:
    python ingestion_pipeline/rebuild_csv.py

Scans OUTPUT_DIR for all *.json files, resolves the source PDF from known raw dirs,
then rewrites parsing_progress.csv with full link columns (pdf_link, json_link).
Run manually or call after bulk processing to sync the sheet.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingestion_pipeline.process_syllabi import (
    OUTPUT_DIR,
    SYLLABI_DIR,
    build_progress_row,
    write_progress_csv,
)

RAW_ROOT = SYLLABI_DIR.parent  # tamu_data/raw/

FALLBACK_DIRS = sorted(RAW_ROOT.glob("simple_syllabus_*/"), reverse=True)  # newest first


def find_pdf(stem: str) -> Path:
    """Resolve source PDF for a given file stem, primary dir first then fallbacks."""
    primary = SYLLABI_DIR / f"{stem}.pdf"
    if primary.exists():
        return primary
    for d in FALLBACK_DIRS:
        candidate = d / f"{stem}.pdf"
        if candidate.exists():
            return candidate
    # Return a plausible path even if not found (links will be broken but row still written)
    return SYLLABI_DIR / f"{stem}.pdf"


def main() -> None:
    json_files = sorted(OUTPUT_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {OUTPUT_DIR}")
        return

    rows = []
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            result = json.load(f)
        pdf_path = find_pdf(jf.stem)
        row = build_progress_row(pdf_path, result)
        rows.append(row)
        status = row["status"]
        pdf_found = "✓" if pdf_path.exists() else "✗ (PDF not found)"
        print(f"  {jf.stem:<45} {status}  pdf={pdf_found}")

    write_progress_csv(rows)
    print(f"\nRebuilt {len(rows)} rows -> {OUTPUT_DIR / 'parsing_progress.csv'}")


if __name__ == "__main__":
    main()
