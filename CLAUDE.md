# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
# Activate venv first
source .venv/Scripts/activate  # Git Bash on Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit chat UI
streamlit run app.py
```

## Running the Data Pipeline

All pipeline stages run from the **repo root**. The full sequence to refresh data is:

```bash
# 1. Scrape academic catalog (catalog.tamu.edu) — resumes automatically
make scrape-catalog
# Scrapy crawl: tamu_data/scraper/ → output: tamu_data/raw/catalog/scraped_content.jsonl

# 2. Scrape Spring 2026 course sections + download syllabi (howdyportal.tamu.edu)
make scrape-classes
# Scrapy crawl: tamu_data/scraper/ → output: tamu_data/raw/syllabi/*.pdf

# 3. Parse syllabus PDFs with Gemini 2.5 Flash → structured JSON
GOOGLE_API_KEY=... python process_syllabi.py
# Options: --department CSCE (single dept), --retry-errors (retry failures)
# Output: tamu_data/processed/gemini_parsed/*.json
# Logs:   tamu_data/logs/errors.jsonl, progress.jsonl
# RESUMES AUTOMATICALLY from last position (reads progress.jsonl)

# 4. Create MongoDB Atlas indexes
python -m db.setup_atlas

# 5. Ingest parsed JSONs into MongoDB (embed with Voyage AI + upsert)
python -m db.ingest                     # all departments
python -m db.ingest --department CSCE   # single department
python -m db.ingest --dry-run           # preview without writing
```

To restart the catalog crawl from scratch, delete `tamu_data/scraper/logs/progress_log.txt`.

## Architecture

### Directory Layout

```
tamubot/
├── app.py                     # Streamlit chat UI (3-stage RAG pipeline or Vertex legacy)
├── config.py                  # Env-based config (MongoDB, Voyage, Gemini, GCP)
├── process_syllabi.py         # Production Gemini PDF parser (resume-capable)
├── test_gemini_parse.py       # Test script for Gemini parsing (sample PDFs)
├── db/                        # MongoDB Atlas + RAG pipeline
│   ├── models.py              # Pydantic v2 models (ChunkDoc, PolicyDoc, CourseDoc)
│   ├── setup_atlas.py         # Creates indexes (vector, text, compound metadata)
│   ├── ingest.py              # JSON → MongoDB ingestion with Voyage AI embeddings
│   ├── search.py              # Hybrid search + multi-course retrieval (RRF fusion)
│   ├── router.py              # 8-intent query router + route_retrieve_rerank() orchestrator
│   ├── reranker.py            # Voyage AI rerank-2 cross-encoder reranking
│   └── generator.py           # Outlet LLM — XML context, adaptive prompts, citations
├── tamu_data/
│   ├── raw/
│   │   ├── catalog/           # Scraped catalog JSONL (gitignored)
│   │   └── syllabi/           # Downloaded syllabus PDFs — 7,970 files (gitignored)
│   ├── processed/
│   │   └── gemini_parsed/     # Gemini 2.5 Flash parsed JSON — 259 files (committed)
│   ├── logs/                  # Ingestion progress + error logs (gitignored)
│   └── scraper/               # Scrapy project (catalog + class_search spiders)
├── pipeline/
│   └── legacy/                # Superseded scripts (convert_for_vertex.py, standardize_syllabi.py)
├── scripts/                   # One-off analysis, debug, and scraping exploration scripts
├── research_prompts.md        # Gemini Deep Research prompts
├── Makefile                   # Pipeline automation
└── requirements.txt
```

### Data Flow

```
catalog.tamu.edu          howdyportal.tamu.edu
       │                         │
  catalog spider           class_search spider
       │                         │
tamu_data/raw/catalog/    tamu_data/raw/syllabi/*.pdf
                                 │
                        process_syllabi.py (Gemini 2.5 Flash multimodal)
                                 │
                  tamu_data/processed/gemini_parsed/*.json
                  (structured: 11 categories, metadata, tables)
                                 │
                        db/ingest.py (Voyage AI voyage-3 embeddings)
                                 │
                         MongoDB Atlas (3 collections)
                        ┌────────┼────────┐
                     chunks   courses  policies
                        └────────┼────────┘
                                 │
              ┌──── 3-Stage RAG Pipeline ────┐
              │                              │
   [Stage 1] db/router.py                    │
     Gemini 2.5 Flash: 8-intent              │
     classification, multi-course            │
     extraction, query rewriting             │
              │                              │
   [Retrieval] db/search.py                  │
     Over-retrieve k=20 candidates           │
     Multi-course parallel retrieval         │
              │                              │
   [Stage 2] db/reranker.py                  │
     Voyage AI rerank-2 cross-encoder        │
     Balanced multi-course reranking         │
              │                              │
   [Stage 3] db/generator.py                 │
     Gemini 2.0 Flash: XML context,          │
     adaptive prompts, [Source N] citations  │
              └──────────────────────────────┘
                                 │
                           app.py (Streamlit UI)
```

### App Layer (`app.py` + `config.py`)

- **Streamlit** frontend with persistent chat history in `st.session_state`.
- **Two backends** controlled by `RETRIEVAL_BACKEND` env var:
  - `mongodb` (default): 3-stage pipeline via `db/router.py` → `db/search.py` + `db/reranker.py` → `db/generator.py`. Uses `google-genai` SDK.
  - `vertex` (legacy): Uses `VertexRagRetriever` → Vertex AI Managed Spanner Corpus → ChatVertexAI.
- Config loaded from `.env` via `python-dotenv`; see `.env.example` for all variables.
- **Two Gemini models**: `MODEL_NAME` (gemini-2.5-flash) for router, `GENERATION_MODEL` (gemini-2.0-flash) for generator. Separated because 2.5-flash has a markdown-table whitespace bug (see Known Issues).

### MongoDB Collections (`db/models.py`)

- **chunks**: Denormalized syllabus chunks. Fields: `crn`, `chunk_index`, `category`, `title`, `content`, `course_id`, `section`, `term`, `instructor_name`, `embedding` (1024-dim Voyage AI). Keyed by `(crn, chunk_index)` for idempotent upserts.
- **policies**: Deduplicated university boilerplate policies. Keyed by SHA-256 hash of policy name. Tracks which CRNs reference each policy.
- **courses**: One doc per section (CRN). Full metadata for aggregate queries (how many sections, which instructors, etc.).

### Search Functions (`db/search.py`)

| Function | Use case | Method |
|---|---|---|
| `hybrid_search(query, filters, k)` | General questions | Manual RRF of `$vectorSearch` + `$search` |
| `multi_course_retrieve(query, course_ids, category, k_per_course)` | Comparison queries | Parallel filtered hybrid searches per course |
| `search_semantic(query, filters, top_k)` | Pure similarity | `$vectorSearch` only |
| `search_by_course(course_id, category)` | Direct lookup | Metadata filter (no embedding) |
| `get_policy(policy_name)` | Policy lookup | Regex match on policy name |
| `aggregate_query(category, course_id)` | Counts/comparisons | `$group` aggregation on courses collection |

### Query Router (`db/router.py`)

Single Gemini 2.5 Flash call classifies user intent → structured JSON. The `route_retrieve_rerank()` function orchestrates the full Stage 1+2 pipeline: classify → retrieve → rerank.

**8 Intent Types**: `single_course_lookup`, `multi_course_comparison`, `aggregation_query`, `policy_lookup`, `schedule_query`, `instructor_query`, `general_academic`, `out_of_scope`

**RouterResult dataclass**: `intent`, `course_ids: list[str]`, `section`, `category`, `policy_name`, `rewritten_query`, `confidence`, `requires_retrieval`, `is_comparison`

Falls back to `hybrid_search` when confidence < 0.5. Comparison queries route to `multi_course_retrieve()` + `rerank_multi_course()`.

### Reranker (`db/reranker.py`)

Voyage AI rerank-2 cross-encoder reranking:
- `rerank(query, documents, top_k)` — general reranking of over-retrieved candidates
- `rerank_multi_course(query, course_groups, top_k_per_course)` — balanced per-course reranking with round-robin interleaving for comparison queries

### Generator (`db/generator.py`)

Outlet LLM (Gemini 2.0 Flash):
- `format_context_xml(results)` — XML-tagged chunks with source/course/category attributes
- `build_system_prompt(intent, course_ids)` — intent-adaptive system prompts
- `generate(results, question, intent, course_ids)` — grounded generation with `[Source N]` citations
- `out_of_scope` intent returns a canned response (no LLM call)
- Post-processing: collapses excessive whitespace from Gemini markdown table generation

### Syllabus Parsing (`process_syllabi.py`)

Sends each PDF directly to Gemini 2.5 Flash (multimodal) and gets structured JSON back in one API call. Each JSON contains:
- `course_metadata` — course_id, section, term, CRN, instructor, TAs, meeting times, location
- `chunks[]` — each with `category` (one of 11 types), `title`, `content` (tables as Markdown), `has_table`
- `boilerplate_policies[]` — names only (full text stored once as golden copy)
- `completeness_check` — missing_sections, warnings

**11 semantic categories**: COURSE_OVERVIEW, INSTRUCTOR, PREREQUISITES, LEARNING_OUTCOMES, MATERIALS, GRADING, SCHEDULE, ATTENDANCE_AND_MAKEUP, AI_POLICY, UNIVERSITY_POLICIES, SUPPORT_SERVICES

Features: resume from last position, error logging, rate limit handling, `--retry-errors` flag.

## Current Status (as of 2026-02-23)

### Completed
- Scrapy spiders for catalog + class search (all departments)
- Syllabus PDF download — 7,970 PDFs across all departments
- **Gemini PDF parsing** — 259/259 CSCE+ISEN files parsed (74 errors retried)
- MongoDB Atlas integration layer: models, indexes, ingestion, search (db/ package)
- **3-stage RAG pipeline**: Router (8 intents, multi-course, query rewriting) → Retrieval+Rerank (Voyage rerank-2) → Generator (XML context, citations)
- **SDK migration**: `google-generativeai` → `google-genai>=1.0`
- App rewired to 3-stage pipeline; single-course, instructor, schedule, aggregation, policy, out-of-scope queries all working end-to-end
- Project restructured: clean root, `tamu_data/raw/` + `tamu_data/processed/`, legacy data deleted
- **Observability stack (Phase 1)** — Langfuse tracing + RAGAS automated evaluation live
  - `db/observability.py`: custom REST-based Langfuse client (bypasses SDK for Python 3.14 compat)
  - Full 5-span trace hierarchy per request (Router → Retrieval → [Embeddings, Search, Reranker] → Generator)
  - RAGAS Faithfulness + AnswerRelevancy scored asynchronously using Voyage AI embeddings as critic
  - Langfuse project: https://cloud.langfuse.com/project/cmlyjfvy200qbad07ezy65y21

### Known Issues
- **Comparison query whitespace bug**: Gemini generates markdown table cells padded with thousands of spaces when receiving near-duplicate chunks (e.g. multiple sections of same course). Post-processing collapses whitespace but the table is truncated because tokens were wasted.
  - **Root cause**: Retrieved chunks for "compare CSCE 120 and CSCE 221" are near-duplicates — 3 CSCE 120 chunks + 3 CSCE 221 chunks with identical content across sections.
  - **Likely fixes**:
    1. **Deduplicate before generation** — keep only one chunk per `(course_id, category)` before passing to generator
    2. **Prompt engineering** — instruct model to use bullet points instead of tables
- **Langfuse SDK incompatible with Python 3.14**: Official SDK uses `pydantic.v1` which breaks on Python 3.14+. Workaround: custom `MinimalLangfuseClient` in `db/observability.py` posts directly to REST API. Revert to official SDK when they ship a fix.

### Next Steps
1. **Fix comparison query whitespace bug** — deduplicate chunks by `(course_id, category)` before generator
2. **Expand parsing to all departments** — run `process_syllabi.py` without `--department` filter
3. **Add latency percentile tracking** — p50/p95 per intent type in Langfuse dashboards
4. **Set score alerts** — Langfuse webhook when faithfulness < 0.5
5. **Observability Phase 2** — prompt management via Langfuse, A/B testing router prompts

## Cloud Deployment

- **Service:** Cloud Run (`tamu-bot-service`, `us-central1`)
- **URL:** `https://tamu-bot-service-653181891130.us-central1.run.app`
- **RAG Corpus (legacy):** `projects/glossy-surge-486017-g8/locations/us-south1/ragCorpora/2305843009213693952`
