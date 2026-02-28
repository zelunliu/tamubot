# CLAUDE.md

> Module-level detail: `rag/CLAUDE.md`, `ingestion_pipeline/CLAUDE.md`, `evals/CLAUDE.md`

## Commands

```bash
source .venv/Scripts/activate && streamlit run app.py   # Windows Git Bash
make test | lint | typecheck | format | eval-router
```

Pipeline (always from repo root):
```bash
make scrape-catalog
make scrape-classes
GOOGLE_API_KEY=... python ingestion_pipeline/process_syllabi.py [--department CSCE] [--retry-errors]
python -m rag.setup_atlas
python -m rag.ingest [--department CSCE] [--dry-run]
```
Reset catalog crawl: delete `tamu_data/scraper/logs/progress_log.txt`

## Gotchas

- **Config**: always `import config` in `rag/` — never `os.getenv()` directly
- **TAMU AI gateway** (`TAMU_API_KEY` set → `USE_TAMU_API=True`): always returns SSE regardless of `stream` param → ALL calls must use `stream=True` + `"".join(chunk.choices[0].delta.content or "" for chunk in stream)`. Base URL: `https://chat-api.tamu.ai/openai` (no `/v1`). Min `max_tokens=4096` or response is empty.
- **Gemini JSON mode**: with `response_mime_type="application/json"` + schema, free-form Markdown fields silently return empty — always render Markdown in Python from structured data
- **Langfuse SDK / Python 3.14**: `pydantic.v1` incompatible → custom `MinimalLangfuseClient` in `rag/observability.py`; revert when SDK ships fix
- **`get_missing_sections()`**: unreliable — `categories_present` derived from ingested chunks, not parser's `completeness_check`. Fix: store `missing_sections` on course doc at ingest time
- **ingestion_pipeline**: stays on direct `GOOGLE_API_KEY` — PDF multimodal (`Part.from_bytes`) not supported by TAMU gateway

## Known Issues

- **Recall@k 36%**: CRN-exact matching counts cross-section hits as misses → redefine hit as `course_id + category`
- **Golden set ~10 label errors**: run adjudication before trusting router accuracy (74% raw, ~90% estimated)
- **Router token budget**: `thinking_budget=512` + `max_output_tokens=1024` — watch if prompt grows
- **pyOpenSSL + cryptography 46.x**: negative X.509 serial numbers → pymongo deprecation warnings; becomes hard error in future release
