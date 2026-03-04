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


```

