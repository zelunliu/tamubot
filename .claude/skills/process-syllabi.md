# /process-syllabi — Run the syllabus processing pipeline

Parse syllabi with Gemini and inspect the results. Supports three modes:
- **Test specific courses** — e.g. `/process-syllabi CSCE 638 670`
- **Full department** — e.g. `/process-syllabi --department CSCE`
- **All departments** — e.g. `/process-syllabi --all`

`GOOGLE_API_KEY` is read automatically from `.env`.

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

1. Glob `tamu_data/raw/syllabi/202611_<DEPT>_<NUM>_*.pdf` for each course.
2. Delete existing JSONs in `tamu_data/processed/gem_parsed_<today>/` for those stems.
3. Run in background:

```bash
cd /c/dev/TAMU_NEW && source .venv/Scripts/activate && \
GOOGLE_API_KEY=$(grep GOOGLE_API_KEY .env | cut -d= -f2) PYTHONIOENCODING=utf-8 python - <<'PYEOF'
import os, json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path.cwd()))
from google import genai
from ingestion_pipeline.process_syllabi import (
    parse_pdf, OUTPUT_DIR, write_per_file_report,
    build_progress_row, load_progress_csv, write_progress_csv, append_progress_jsonl,
)

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
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

After completion, display the COURSE_SUMMARY for each course plus any flags.

---

## Step 2b — Department or all-departments mode

Run the main script directly (handles skipping, rate limits, progress CSV, per-file reports):

```bash
cd /c/dev/TAMU_NEW && source .venv/Scripts/activate && \
GOOGLE_API_KEY=$(grep GOOGLE_API_KEY .env | cut -d= -f2) \
python ingestion_pipeline/process_syllabi.py [--department DEPT]
```

Run in background. Stream / tail output if the user wants live progress.

---

## Step 3 — Report

**Targeted mode:** show the COURSE_SUMMARY content and flags for each course.
**Department/all mode:** summarize final counts (succeeded / failed) from stdout.

---

## Notes

- **Close the progress CSV before running** — it is rewritten after every file; Excel locks cause PermissionError.
- OUTPUT_DIR is date-stamped (`gem_parsed_YYYYMMDD`); new folder each calendar day.
- To force a re-parse of already-done files, delete their JSONs from the output dir first.

## Examples

```
/process-syllabi CSCE 670
/process-syllabi CSCE 638 670
/process-syllabi --department CSCE
/process-syllabi --all
```
