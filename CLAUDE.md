# CLAUDE.md

## Task Notification

Timer starts automatically via `UserPromptSubmit` hook. Notification fires via `Stop` hook for tasks `>15s`. No manual tool calls needed.

> Module-level detail: `rag/CLAUDE.md`, `ingestion_pipeline/CLAUDE.md`, `evals/CLAUDE.md`

## Commands

```bash
source .venv/Scripts/activate && streamlit run app.py   # Windows Git Bash
make test | lint | typecheck | format | eval-router | probe | probe-full
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

Benchmarking:
```bash
make eval-draft                      # generate Excel draft from corpus CRNs
make import-draft DRAFT=<path>       # approved Excel → JSONL golden set
make bench GOLDEN=<path> EXP=<name>  # run benchmark → Excel + MD reports
make bench-ragas GOLDEN=<path> EXP=<name>   # with RAGAS scores
make validate-ragas BENCH=<path>     # RAGAS vs human judgment correlation
```
Reports: `tamu_data/evals/reports/benchmark_{EXP}_{YYYYMMDD}.xlsx` (tabs: Summary / Per-Query / Config)

## Gotchas

- **Config**: always `import config` in `rag/` — never `os.getenv()` directly
- **TAMU AI gateway** (`TAMU_API_KEY` set → `USE_TAMU_API=True`): always returns SSE regardless of `stream` param → ALL calls must use `stream=True` + `"".join(chunk.choices[0].delta.content or "" for chunk in stream)`. Base URL: `https://chat-api.tamu.ai/openai` (no `/v1`). Min `max_tokens=4096` or response is empty.
- **Gemini JSON mode**: with `response_mime_type="application/json"` + schema, free-form Markdown fields silently return empty — always render Markdown in Python from structured data
- **Langfuse SDK / Python 3.14**: `pydantic.v1` incompatible → custom `MinimalLangfuseClient` in `rag/observability.py`; revert when SDK ships fix
- **ingestion_pipeline**: stays on direct `GOOGLE_API_KEY` — PDF multimodal (`Part.from_bytes`) not supported by TAMU gateway

## Skills — Auto-Engage

Invoke via the Skill tool automatically (no `/` command needed) when intent matches:

- **probe-rag**: user asks to test a query, run a probe, check RAG output, or inspect a Langfuse trace
- **scrape**: user asks to scrape a site, download syllabi, or add/run a crawler
- **process-syllabi**: user asks to parse/process syllabi or run the ingestion pipeline on PDFs
- **github-collab**: user says "push", "open a PR", "create a branch", "start a feature", "I merged", "clean up branch", "am I ready to push", or "run checks"
- **refine-syllabi**: user asks to audit, refine, or improve syllabus parsing quality, check for boilerplate leaks, or iterate on the ingestion prompt

When skill tool engaged, make sure to notify user!

