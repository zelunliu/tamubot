# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.
For module-level detail, see the CLAUDE.md in each subdirectory.

> **Maintenance**: Update this file (Recent Work, App Layer, Known Issues) after any architectural change, new dependency, or config addition. Update the relevant subdirectory CLAUDE.md too.

## Running the App

```bash
source .venv/Scripts/activate  # Git Bash on Windows
pip install -r requirements.txt
streamlit run app.py
```

## Running the Data Pipeline

All stages run from the **repo root**:

```bash
# 1. Scrape academic catalog
make scrape-catalog
# → tamu_data/raw/catalog/scraped_content.jsonl

# 2. Scrape course sections + download syllabi
make scrape-classes
# → tamu_data/raw/syllabi/*.pdf

# 3. Parse syllabus PDFs → structured JSON (resumes automatically)
GOOGLE_API_KEY=... python ingestion_pipeline/process_syllabi.py
# --department CSCE   (single dept)   --retry-errors  (retry failures)
# → tamu_data/processed/gemini_parsed/*.json

# 4. Create MongoDB Atlas indexes
python -m rag.setup_atlas

# 5. Ingest into MongoDB (Voyage AI embeddings + upsert)
python -m rag.ingest                      # all departments
python -m rag.ingest --department CSCE    # single department
python -m rag.ingest --dry-run            # preview without writing
```

To restart the catalog crawl from scratch: delete `tamu_data/scraper/logs/progress_log.txt`.

## Dev Commands

```bash
make test        # pytest tests/ -v
make lint        # ruff check rag/ app.py config.py
make typecheck   # mypy rag/ --ignore-missing-imports
make format      # ruff format rag/ app.py config.py
make eval-router # python evals/eval_router_metrics.py
```

## Architecture

### Directory Layout

```
tamubot/
├── app.py                        # Streamlit chat UI
├── config.py                     # Env-based config — all secrets/constants live here
├── rag/                          # 3-stage RAG pipeline (see rag/CLAUDE.md)
│   ├── router.py                 # Stage 1: query parsing + retrieval orchestration
│   ├── search.py                 # MongoDB Atlas hybrid/vector/metadata search
│   ├── reranker.py               # Voyage AI rerank-2 cross-encoder
│   ├── generator.py              # Stage 3: generate_stream(), generate_comparison()
│   ├── prompts.py                # All LLM prompt strings + temperatures
│   ├── context_builder.py        # format_context_xml() — primacy-recency assembly
│   ├── gates.py                  # Gate 1 citation validation
│   ├── models.py                 # Pydantic v2 models (ChunkDoc, PolicyDoc, CourseDoc)
│   ├── comparison_schemas.py     # CourseComparisonTable schema
│   ├── observability.py          # Langfuse tracing + RAGAS + Gate 2 groundedness
│   ├── setup_atlas.py            # Creates MongoDB indexes
│   └── ingest.py                 # JSON → MongoDB ingestion
├── ingestion_pipeline/           # Syllabus data pipeline (see ingestion_pipeline/CLAUDE.md)
│   ├── process_syllabi.py        # Gemini 2.5 Flash PDF parser (resume-capable)
│   └── legacy/                   # Superseded scripts
├── evals/                        # Eval framework (see evals/CLAUDE.md)
├── tests/                        # Unit tests (see tests/CLAUDE.md)
├── scripts/                      # One-off analysis artifacts
├── docs/                         # Reference docs (OBSERVABILITY.md, PROJECT_CONTEXT.md)
├── tamu_data/
│   ├── raw/catalog/              # Scraped catalog JSONL (gitignored)
│   ├── raw/syllabi/              # Downloaded PDFs — 7,970 files (gitignored)
│   ├── processed/gemini_parsed/  # Parsed JSON — 259 files (committed)
│   ├── logs/                     # errors.jsonl, progress.jsonl, ingest_run.log
│   ├── evals/                    # Golden sets + eval reports (gitignored)
│   └── scraper/                  # Scrapy project (catalog + class_search spiders)
├── Makefile
└── requirements.txt
```

### Data Flow

```
catalog.tamu.edu       howdyportal.tamu.edu
       │                       │
  catalog spider         class_search spider
       │                       │
tamu_data/raw/         tamu_data/raw/syllabi/*.pdf
                               │
              ingestion_pipeline/process_syllabi.py
              (Gemini 2.5 Flash multimodal → 11 categories)
                               │
              tamu_data/processed/gemini_parsed/*.json
                               │
              rag/ingest.py (Voyage AI voyage-3 embeddings)
                               │
                       MongoDB Atlas
                    chunks | courses | policies
                               │
            ┌─── 3-Stage RAG Pipeline (rag/) ───┐
            │                                    │
     [Stage 1] router.py                         │
       Gemini 2.5 Flash: variable extraction     │
       + pure-Python function derivation         │
            │                                    │
     [Stage 2] search.py + reranker.py           │
       Hybrid RRF retrieval → Voyage rerank-2    │
            │                                    │
     [Stage 3] generator.py                      │
       Gemini 2.5 Flash: primacy-recency XML,    │
       citations, Gate 1 + Gate 2 scoring        │
            └────────────────────────────────────┘
                               │
                         app.py (Streamlit)
```

### App Layer

- **Backend**: `RETRIEVAL_BACKEND=mongodb` (default) uses the 3-stage `rag/` pipeline.
  `vertex` (legacy) uses Vertex AI Managed Spanner Corpus.
- **Config**: all env vars loaded from `.env` via `config.py`. Never call `os.getenv()` directly
  in `rag/` — always `import config`.
- **Models**: `MODEL_NAME` = router, `GENERATION_MODEL` = generator (both `gemini-2.5-flash`),
  `VALIDATION_MODEL` = Gate 2 critic (`gemini-2.5-flash-lite`).
- **TAMU AI API**: when `TAMU_API_KEY` is set, `USE_TAMU_API=True` and all `rag/` LLM calls route
  through `https://chat-api.tamu.ai/openai` (`protected.gemini-2.5-flash`) via the `openai` SDK.
  `ingestion_pipeline/process_syllabi.py` stays on direct `GOOGLE_API_KEY` (PDF multimodal).
  Gateway quirk: always streams SSE — all calls use `stream=True` + chunk accumulation.

## Current Status (as of 2026-02-27)

### Pipeline State
- **Parsing**: 259/259 CSCE+ISEN syllabi parsed (7,970 PDFs total downloaded)
- **MongoDB**: fully ingested; 3 collections (chunks, courses, policies) with vector + text indexes
- **Router accuracy**: 74% on unadjudicated golden set (~90% estimated after label correction)
- **Recall@k**: 36% on first run (known issue — see below)
- **Citation rate**: 75%; 0 pipeline errors

### Recent Work
- **2026-02-27**: TAMU AI API integration — all RAG LLM calls routed through institutional
  OpenAI-compatible gateway (`protected.gemini-2.5-flash`); `openai>=1.0` added
- **2026-02-27**: Module split (`rag/prompts.py`, `context_builder.py`, `gates.py`); directory
  renames (`db/` → `rag/`, `pipeline/` → `ingestion_pipeline/`); per-directory CLAUDE.md files
- **2026-02-25**: Generator — primacy-recency bracketing, single-call comparison, streaming,
  Gate 1/Gate 2 validation, thinking budget 4096→1024
- **2026-02-23**: Router — `ADMINISTRATIVE` semantic type, `*_combined` always hybrid,
  background-course suppression
- **Earlier**: Observability stack (Langfuse + RAGAS), eval framework (golden set, router metrics,
  pipeline eval, adjudication), new router schema (variable extraction → pure-Python derivation)

## Known Issues

- **Recall@k 36%**: CRN-exact matching counts cross-section hits as misses. Fix: redefine hit as
  `course_id + category` match. Needs per-function breakdown.
- **Golden set ~10 label errors**: mechanical label assignment from strata. Run
  `evals/adjudicate_golden_set.py` before trusting accuracy numbers.
- **Router token budget**: `thinking_budget=512` + `max_output_tokens=1024`. Watch if prompt grows.
- **Langfuse SDK / Python 3.14**: `pydantic.v1` incompatibility. Workaround: custom
  `MinimalLangfuseClient` in `rag/observability.py`. Revert when SDK ships a fix.
- **`get_missing_sections()` unreliable**: `categories_present` derived from ingested chunks, not
  parser's `completeness_check`. Fix: store `missing_sections` directly on course doc at ingest time.
- **pyOpenSSL + cryptography 46.x**: negative X.509 serial numbers trigger deprecation warnings
  from pymongo. Will become hard error in future cryptography release.

## Next Steps

1. **Adjudicate golden set** — `python evals/adjudicate_golden_set.py --golden-set tamu_data/logs/golden_set.jsonl --router-results tamu_data/logs/router_metrics.json --output tamu_data/logs/golden_set_v2.jsonl`
2. **Re-run eval on adjudicated set** — measure true router accuracy; add `--ragas` flag
3. **Fix recall@k** — redefine hit as `course_id + category`; add per-function breakdown
4. **Expand parsing** — run `ingestion_pipeline/process_syllabi.py` without `--department` filter
5. **Latency tracking** — p50/p95 per function type in Langfuse dashboards
6. **Observability Phase 2** — prompt management via Langfuse, A/B testing router variables

## Cloud Deployment

- **Service:** Cloud Run (`tamu-bot-service`, `us-central1`)
- **URL:** `https://tamu-bot-service-653181891130.us-central1.run.app`
- **RAG Corpus (legacy):** `projects/glossy-surge-486017-g8/locations/us-south1/ragCorpora/2305843009213693952`
