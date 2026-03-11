# ingestion_pipeline/

## CBD Rationale

`ingestion_pipeline/` is the **producer**; `rag.models` is the **contract** it implements. Import schema models from `rag.models`, never define them here.

## Running (V3 pipeline)

```bash
python ingestion_pipeline/process_syllabi_v3.py --pilot
python ingestion_pipeline/process_syllabi_v3.py --pdf tamu_data/raw/simple_syllabus_20260305/<file>.pdf [--new-run] [--force]
python ingestion_pipeline/process_syllabi_v3.py --department CSCE
python -m ingestion_pipeline.ingest [--department CSCE] [--dry-run]
```

Pipeline (always from repo root):
```bash
make scrape-catalog
make scrape-classes
GOOGLE_API_KEY=... python ingestion_pipeline/process_syllabi.py [--department CSCE] [--retry-errors]
python -m ingestion_pipeline.setup_atlas
python -m ingestion_pipeline.ingest [--department CSCE] [--dry-run]
python -m ingestion_pipeline.ingest --crns-file tamu_data/evals/eval_corpus.json  # corpus only
```
Reset catalog crawl: delete `tamu_data/scraper/logs/progress_log.txt`

**Always run all steps (0–3) together.** `--step N` is for debugging only — partial runs leave downstream outputs stale.

## Gotchas

- **Uses TAMU gateway** for step 3 LLM calls (`config.get_tamu_client()`). PyMuPDF extracts PDFs directly — no `Part.from_bytes`.
- **`\ufffd` chars**: PyMuPDF emits U+FFFD for un-decodable bytes. `clean_replacement_chars()` replaces with `-` post-parse.
- **Boilerplate registry** (`boilerplate_stripper.py`): font-annotated headers matched by `BOILERPLATE_REGISTRY`; body-size headers matched by `BODY_BOILERPLATE_HEADERS` + `strip_body_level_boilerplate()`. Only add long, unambiguous phrases to body list.
- **`_BP_KEYWORDS`** in `process_syllabi_v3.py`: flags non-stripped headers as new candidates → `new_bp_candidates` column in combined log. Expand when new boilerplate patterns emerge.
- **Legacy `process_syllabi.py`**: not used in v3. Still functional for Gemini semantic extraction if needed.

