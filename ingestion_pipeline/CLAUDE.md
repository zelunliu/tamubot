# ingestion_pipeline/

## CBD Rationale

`ingestion_pipeline/` is the **producer**; `rag.models` is the **contract** it implements. Import schema models from `rag.models`, never define them here.

```python
from ingestion_pipeline import parse_pdf, run_ingest, setup_indexes
```

## Gotchas

- **Always uses `GOOGLE_API_KEY` directly** — not the TAMU gateway. PDF bytes (`Part.from_bytes`) are not supported by OpenAI-compatible APIs.
- **13 vs 11 categories**: `process_syllabi.py` uses `ALL_CATEGORIES` (13, incl. `COURSE_SUMMARY` + `SAFETY`); `rag.models.VALID_CATEGORIES` has 11 — intentional, `COURSE_SUMMARY`/`SAFETY` not query-routable yet.
- **Output folder is date-stamped**: `gem_parsed_YYYYMMDD` — new folder each calendar day. `OUTPUT_DIR` is computed at import time via `datetime.now()`.
- **CSV schema**: `CSV_FIELDS` constant drives all writes (`extrasaction="ignore"`, `restval=""`). Delete stale CSV manually when adding new category columns.
- **`\ufffd` replacement chars**: Gemini substitutes U+FFFD for un-decodable PDF bytes (e.g. Windows-1252 en-dashes). `clean_replacement_chars()` replaces with `-` after every successful parse.

## Running

```bash
GOOGLE_API_KEY=... python ingestion_pipeline/process_syllabi.py [--department CSCE] [--retry-errors]
GOOGLE_API_KEY=... python ingestion_pipeline/refine_errors.py [--department CSCE]  # retry error JSONs
python -m ingestion_pipeline.setup_atlas
python -m ingestion_pipeline.ingest [--department CSCE] [--dry-run]
```

Resume: skips already-parsed files in `OUTPUT_DIR`. Errors logged to `tamu_data/logs/errors.jsonl`.
Per-file reports: `tamu_data/logs/per_file/<stem>.txt` (written for every PDF, success and failure).
Progress sheet: `tamu_data/processed/gem_parsed_YYYYMMDD/parsing_progress.csv` — updates after each file.

## Output Schema

```json
{
  "course_metadata": { "course_id", "section", "term", "crn", "instructor", "tas", "meeting_times", "location", "course_url" },
  "chunks": [{"category": "GRADING", "title": "...", "content": "...", "has_table": false}],
  "boilerplate_policies": [...],
  "completeness_check": {"missing_sections": ["SCHEDULE"], "warnings": [...]},
  "_source_file": "202611_CSCE_670_600_46627.pdf",
  "_parsed_at": "2026-03-04T10:00:00"
}
```

## 13 Categories

`COURSE_OVERVIEW` · `INSTRUCTOR` · `PREREQUISITES` · `LEARNING_OUTCOMES` · `MATERIALS` · `GRADING` · `SCHEDULE` · `ATTENDANCE_AND_MAKEUP` · `AI_POLICY` · `UNIVERSITY_POLICIES` · `SUPPORT_SERVICES` · `COURSE_SUMMARY` · `SAFETY`

**`COURSE_SUMMARY`** — always generated; RAG keyword-dense format (Topics / Methods / Prerequisites / Tools / Niche). Target 200–280 tokens. No narrative prose.

**`SAFETY`** — lab/hands-on courses only. PPE, chemical handling, emergency procedures, equipment rules. Never use for generic academic integrity content.
