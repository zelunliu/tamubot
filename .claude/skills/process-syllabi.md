# /process-syllabi — Run the syllabus processing pipeline

Parse syllabi via TAMU API and inspect the results. Supports three modes:
- **Test specific courses** — e.g. `/process-syllabi CSCE 638 670`
- **Full department** — e.g. `/process-syllabi --department CSCE`
- **All departments** — e.g. `/process-syllabi --all`

`TAMU_API_KEY` is read automatically from `.env` via `config`.

---

## Step 1 — Parse arguments

| Input | Mode |
|---|---|
| Course codes (e.g. `CSCE 638 670`) | targeted — delete existing JSONs for those courses, reparse only them |
| `--department DEPT` | full department via `main()` script |
| `--all` | all departments via `main()` script |
| no args | ask the user which mode they want |

---

## Step 2a — Targeted mode (specific courses)

1. Glob `tamu_data/raw/syllabi/202611_<DEPT>_<NUM>_*.pdf` for each course. If not found there, also try `tamu_data/raw/simple_syllabus_*/`.
2. Delete existing JSONs in `tamu_data/processed/gem_parsed_<today>/` for those stems.
3. Run in background:

```bash
cd /c/dev/TAMU_NEW && source .venv/Scripts/activate && PYTHONIOENCODING=utf-8 python - <<'PYEOF'
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path.cwd()))
import config
from ingestion_pipeline.process_syllabi import (
    parse_pdf, OUTPUT_DIR, write_per_file_report,
    build_progress_row, load_progress_csv, write_progress_csv, append_progress_jsonl,
)

client = config.get_tamu_client()
targets = [
    # FILLED IN BY CLAUDE — one Path(...) per PDF
]

progress_rows = load_progress_csv()
progress_index = {r["file"]: i for i, r in enumerate(progress_rows)}

for pdf_path in targets:
    print(f"\n{'='*60}\nParsing {pdf_path.name}...")
    result = parse_pdf(client, pdf_path)
    out_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    write_per_file_report(pdf_path, result)
    row = build_progress_row(pdf_path, result)
    if pdf_path.name in progress_index:
        progress_rows[progress_index[pdf_path.name]] = row
    else:
        progress_index[pdf_path.name] = len(progress_rows)
        progress_rows.append(row)
    write_progress_csv(progress_rows)
    append_progress_jsonl(row)
    if "error" not in result:
        for chunk in result.get("chunks", []):
            if chunk.get("category") == "COURSE_SUMMARY":
                print("\n--- COURSE_SUMMARY ---")
                print(chunk["content"])
                break
        print(f"flags: {row.get('flags') or 'none'}")
        print(f"OK -> {out_path}")
    else:
        print(f"ERROR: {result['error']}")
PYEOF
```

After completion, display the COURSE_SUMMARY for each course plus any flags, then run **Step 2c** for each course.

---

## Step 2c — Compare parsed JSON against source PDF (targeted mode only)

For each parsed course, run this inline script to verify content coverage:

```bash
cd /c/dev/TAMU_NEW && source .venv/Scripts/activate && PYTHONIOENCODING=utf-8 python - <<'PYEOF'
import json, sys
from pathlib import Path
import fitz
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

pdf_path  = Path("FILLED_IN_BY_CLAUDE")   # source PDF
json_path = Path("FILLED_IN_BY_CLAUDE")   # output JSON from Step 2a

# Extract all text from PDF
doc = fitz.open(str(pdf_path))
pdf_pages = [page.get_text() for page in doc]
doc.close()
pdf_text = "\n".join(pdf_pages)
pdf_lines = set(l.strip() for l in pdf_text.splitlines() if len(l.strip()) > 50)

with open(json_path, encoding="utf-8") as f:
    parsed = json.load(f)

# Flatten all chunk content
chunks = parsed.get("chunks", [])
all_chunk_text = "\n".join(c["content"] for c in chunks)
chunk_lines = set(l.strip() for l in all_chunk_text.splitlines() if len(l.strip()) > 50)

# Metadata coverage
meta = parsed.get("course_metadata", {})
inst = meta.get("instructor", {})
print(f"\n=== {json_path.stem} ===")
print(f"PDF pages: {len(pdf_pages)} | PDF text: {len(pdf_text)} chars")
print(f"Chunks: {len(chunks)} | Chunk text: {len(all_chunk_text)} chars")
print(f"Coverage: {len(all_chunk_text)/max(len(pdf_text),1)*100:.0f}% of source chars captured")

print(f"\nMetadata: {meta.get('course_id')} sec {meta.get('section')} | {meta.get('term')} | CRN {meta.get('crn')}")
print(f"Instructor: {inst.get('name')} <{inst.get('email')}> | Office: {inst.get('office')}")
print(f"Meeting: {meta.get('meeting_times')} | Credits: {meta.get('credit_hours')}")

print(f"\nCategories:")
for c in chunks:
    print(f"  {c['category']:<25} {len(c['content']):>5}c {'[table]' if c.get('has_table') else ''}")

bp = parsed.get("boilerplate_policies", [])
if bp:
    print(f"\nBoilerplate ({len(bp)}): {', '.join(bp)}")

cc = parsed.get("completeness_check", {})
if cc.get("missing_sections"):
    print(f"\nMissing sections: {', '.join(cc['missing_sections'])}")
if cc.get("warnings"):
    print("Warnings:")
    for w in cc["warnings"]: print(f"  - {w}")

# Lines from PDF not found in any chunk (potential losses)
missing_lines = pdf_lines - chunk_lines
# Filter out lines that are in boilerplate_policies (intentionally excluded)
bp_text = " ".join(bp).lower()
missing_lines = [l for l in missing_lines if not any(word in bp_text for word in l.lower().split()[:3])]
if missing_lines:
    print(f"\nPDF lines NOT captured in chunks ({len(missing_lines)} lines >50c):")
    for l in sorted(missing_lines)[:15]:
        print(f"  ? {l[:120]}")
else:
    print("\nAll significant PDF lines are captured in chunks.")

print(f"\nCOURSE_SUMMARY:")
for c in chunks:
    if c["category"] == "COURSE_SUMMARY":
        print(c["content"])
PYEOF
```

Report what was found: coverage %, any uncaptured lines, and whether the completeness check flags look correct.

---

## Step 2b — Department or all-departments mode

Run the main script directly (handles skipping, rate limits, progress CSV, per-file reports):

```bash
cd /c/dev/TAMU_NEW && source .venv/Scripts/activate && \
python ingestion_pipeline/process_syllabi.py [--department DEPT]
```

Run in background. Stream / tail output if the user wants live progress. No comparison step in this mode.

---

## Step 3 — Report

**Targeted mode:** show the COURSE_SUMMARY content and flags for each course.
**Department/all mode:** summarize final counts (succeeded / failed) from stdout.

---

## Notes

- **Close the progress CSV before running** — it is rewritten after every file; Excel locks cause PermissionError.
- OUTPUT_DIR is date-stamped (`gem_parsed_YYYYMMDD`); new folder each calendar day.
- To force a re-parse of already-done files, delete their JSONs from the output dir first.
- **PDF source dirs**: primary is `tamu_data/raw/syllabi/`; fallback is `tamu_data/raw/simple_syllabus_<date>/`.
- **Coverage check caveat**: `UNIVERSITY_POLICIES` boilerplate text is intentionally excluded from chunks (names only in `boilerplate_policies`) — uncaptured lines from those pages are expected and not a bug.
- **Text extraction**: PyMuPDF extracts embedded text only. Scanned/image-only PDFs will produce near-empty text and poor parse results — check page char counts if output looks thin.

## Examples

```
/process-syllabi CSCE 670
/process-syllabi CSCE 638 670
/process-syllabi --department CSCE
/process-syllabi --all
```
