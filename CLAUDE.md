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
GOOGLE_API_KEY=... python pipeline/process_syllabi.py
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
├── db/                        # MongoDB Atlas + RAG pipeline
│   ├── models.py              # Pydantic v2 models (ChunkDoc, PolicyDoc, CourseDoc)
│   ├── setup_atlas.py         # Creates indexes (vector, text, compound metadata)
│   ├── ingest.py              # JSON → MongoDB ingestion with Voyage AI embeddings
│   ├── search.py              # Hybrid search + multi-course retrieval (RRF fusion)
│   ├── router.py              # 8-intent query router + route_retrieve_rerank() orchestrator
│   ├── reranker.py            # Voyage AI rerank-2 cross-encoder reranking
│   ├── generator.py           # Outlet LLM — primacy-recency context, two-call comparison, citation gates
│   ├── comparison_schemas.py  # Pydantic schemas for two-call comparison extraction (CourseComparisonTable)
│   └── observability.py       # Langfuse tracing + RAGAS evaluation + Gate 2 groundedness scoring
├── pipeline/                  # Data pipeline stages
│   ├── process_syllabi.py     # Production Gemini PDF parser (resume-capable)
│   ├── test_gemini_parse.py   # Manual test harness for Gemini parsing (sample PDFs)
│   └── legacy/                # Superseded scripts (convert_for_vertex.py, standardize_syllabi.py)
├── evals/                     # Evaluation framework
│   ├── eval_pipeline.py       # End-to-end pipeline eval runner
│   ├── eval_generator_tiered.py
│   ├── eval_retrieval_metrics.py
│   ├── eval_router_metrics.py
│   ├── eval_statistics.py
│   ├── generate_golden_set.py
│   └── evals.txt              # Golden test cases
├── scripts/                   # One-off analysis + scraping exploration
│   └── artifacts/             # Generated data files (JSON reports, sample PDFs, HTML)
├── docs/                      # Reference documentation
│   ├── OBSERVABILITY.md
│   ├── research_prompts.md
│   └── PROJECT_CONTEXT.md
├── tamu_data/
│   ├── raw/
│   │   ├── catalog/           # Scraped catalog JSONL (gitignored)
│   │   └── syllabi/           # Downloaded syllabus PDFs — 7,970 files (gitignored)
│   ├── processed/
│   │   └── gemini_parsed/     # Gemini 2.5 Flash parsed JSON — 259 files (committed)
│   ├── logs/                  # Pipeline logs (errors.jsonl=parse, ingest_errors.jsonl=ingest, progress.jsonl, ingest_run.log)
│   ├── evals/                 # Eval run output (gitignored)
│   │   ├── golden_sets/       # golden_set_*.jsonl
│   │   └── reports/           # eval_report_*.md, eval_results_*.jsonl
│   └── scraper/               # Scrapy project (catalog + class_search spiders)
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
                        pipeline/process_syllabi.py (Gemini 2.5 Flash multimodal)
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
     Gemini 2.5 Flash: primacy-recency XML   │
     context, adaptive prompts, citations,   │
     Gate 1 (regex) + Gate 2 (LLM) scoring  │
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
- **Three Gemini models**: `MODEL_NAME` (gemini-2.5-flash) for router, `GENERATION_MODEL` (gemini-2.5-flash) for generator, `VALIDATION_MODEL` (gemini-2.5-flash-lite) for Gate 2 groundedness scoring.
- **Thinking budgets**: `THINKING_BUDGET_METADATA=0` (metadata_* functions, deterministic), `THINKING_BUDGET_SEMANTIC=1024` (hybrid_* and semantic_general, complex reasoning).
- **Temperature constants**: `TEMP_DETERMINISTIC=0.0` (metadata_*, out_of_scope), `TEMP_SYNTHESIS=0.2` (hybrid_*, semantic_general for advisory reasoning).

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

Single Gemini 2.5 Flash call **extracts structured variables** from the query — no intent classification. The `route_retrieve_rerank()` function orchestrates the full Stage 1+2 pipeline: extract → derive → retrieve → rerank.

**Variables extracted by LLM**: `course_ids`, `specific_categories`, `category_confidence`, `specific_only`, `semantic_intent`, `semantic_type`, `rewritten_query`, `section`

**Derived in pure Python** (no LLM): `function` (one of 8) + `retrieval_mode` (`metadata` | `hybrid` | `semantic`)

**8 Functions**: `metadata_default`, `metadata_specific`, `metadata_combined`, `hybrid_default`, `hybrid_specific`, `hybrid_combined`, `semantic_general`, `out_of_scope`

**RouterResult dataclass**: all extracted variables + derived `function` + `retrieval_mode`, auto-computed in `__post_init__`. `requires_retrieval` property = `bool(course_ids) or semantic_intent`.

`_derive_function()` and `_derive_retrieval_mode()` are pure Python helpers implementing the derivation matrix. `category_confidence >= CATEGORY_CONFIDENCE_THRESHOLD (0.7)` → metadata path (no embedding).

### Reranker (`db/reranker.py`)

Voyage AI rerank-2 cross-encoder reranking:
- `rerank(query, documents, top_k)` — general reranking of over-retrieved candidates
- `rerank_multi_course(query, course_groups, top_k_per_course)` — balanced per-course reranking with round-robin interleaving for comparison queries

### Generator (`db/generator.py`)

Outlet LLM (Gemini 2.5 Flash):
- `format_context_xml(results)` — XML-tagged chunks with **primacy-recency bracketing**: rank 1 at context start, rank 2 at context end, ranks 3–N in the middle. Combats Lost-in-the-Middle attention degradation.
- `build_system_prompt(function, course_ids, semantic_type, category_confidence)` — function-adaptive prompts + semantic_type advisory overlay + **Verbal Uncertainty Calibration** (injects uncertainty language when `category_confidence < 0.7`).
- `generate(...)` — single-course grounded generation with `[Source N]` citations + thinking for hybrid/semantic functions. Routes multi-course queries to `generate_comparison`.
- `generate_stream(...)` — streaming variant of `generate()`: yields text chunks via `generate_content_stream()` for single-course queries; falls back to blocking `generate_comparison()` for multi-course (JSON schema can't stream). Gate 1 + Gate 2 fire post-stream on accumulated text.
- `generate_comparison(results, question, course_ids, trace)` — **single-call architecture** for multi-course comparisons: one Gemini call extracts structured JSON via `CourseComparisonTable` schema; `_render_comparison_markdown()` builds the 4-column table + Detailed Comparison prose in Python. Saves ~5–10s vs former two-call pattern.
- `validate_citations_gate1(response_text)` / `validate_citations_with_trace(...)` — **Gate 1** regex check: verifies at least one `[Source N]` citation is present in factual responses.
- Gate 2 groundedness scoring fires asynchronously via `run_groundedness_scoring_background` after each response (RAGAS `ResponseGroundedness`, Gemini 2.5 Flash-Lite as critic).
- `out_of_scope` returns a canned response (no LLM call).
- Per-function temperatures: `metadata_*` → 0.0, `hybrid_*` → 0.2, `semantic_general` → 0.2.
- Post-processing: collapses excessive whitespace from Gemini markdown table generation.

### Comparison Schemas (`db/comparison_schemas.py`)

Pydantic models for the single-call comparison architecture:
- `CourseComparisonData` — per-course fields: `course_id`, `grading`, `workload`, `prerequisites`, `course_overview`, `learning_outcomes`, `topics_complexity`, `materials`.
- `CourseComparisonTable` — wrapper holding a list of `CourseComparisonData`; used as `response_schema` to enforce structured JSON extraction. Markdown is rendered in Python by `_render_comparison_markdown()` — not by the LLM — because Gemini in JSON mode reliably fills typed fields but not free-form string fields.

### Syllabus Parsing (`pipeline/process_syllabi.py`)

Sends each PDF directly to Gemini 2.5 Flash (multimodal) and gets structured JSON back in one API call. Each JSON contains:
- `course_metadata` — course_id, section, term, CRN, instructor, TAs, meeting times, location
- `chunks[]` — each with `category` (one of 11 types), `title`, `content` (tables as Markdown), `has_table`
- `boilerplate_policies[]` — names only (full text stored once as golden copy)
- `completeness_check` — missing_sections, warnings

**11 semantic categories**: COURSE_OVERVIEW, INSTRUCTOR, PREREQUISITES, LEARNING_OUTCOMES, MATERIALS, GRADING, SCHEDULE, ATTENDANCE_AND_MAKEUP, AI_POLICY, UNIVERSITY_POLICIES, SUPPORT_SERVICES

Features: resume from last position, error logging, rate limit handling, `--retry-errors` flag.

## Current Status (as of 2026-02-25)

### Completed
- Scrapy spiders for catalog + class search (all departments)
- Syllabus PDF download — 7,970 PDFs across all departments
- **Gemini PDF parsing** — 259/259 CSCE+ISEN files parsed (74 errors retried)
- MongoDB Atlas integration layer: models, indexes, ingestion, search (db/ package)
- **3-stage RAG pipeline**: Router → Retrieval+Rerank (Voyage rerank-2) → Generator (XML context, citations)
- **SDK migration**: `google-generativeai` → `google-genai>=1.0`
- Project restructured: clean root, `tamu_data/raw/` + `tamu_data/processed/`, legacy data deleted
- **Observability stack (Phase 1)** — Langfuse tracing + RAGAS automated evaluation live
  - `db/observability.py`: custom REST-based Langfuse client (bypasses SDK for Python 3.14 compat)
  - Full 5-span trace hierarchy per request (Router → Retrieval → [Embeddings, Search, Reranker] → Generator)
  - RAGAS Faithfulness + AnswerRelevancy scored asynchronously using Voyage AI embeddings as critic
  - Langfuse project: https://cloud.langfuse.com/project/cmlyjfvy200qbad07ezy65y21
- **New router schema** — replaced 8-intent classification with structured variable extraction
  - Router LLM extracts: `course_ids`, `specific_categories`, `category_confidence`, `specific_only`, `semantic_intent`, `semantic_type`, `rewritten_query`
  - Function and retrieval_mode derived in pure Python from those variables (no LLM judgment for routing)
  - 8 functions: `metadata_default/specific/combined`, `hybrid_default/specific/combined`, `semantic_general`, `out_of_scope`
  - `search_by_course_categories()` added — exact index lookup, no embedding, no reranking (metadata path)
  - `_deduplicate_chunks()` in place — fixes comparison query whitespace bug
  - **Dry-run eval: 34/34 (100%)** across all function types (CSCE 638 + CSCE 670 test suite)
  - Key routing insight: `hybrid_specific` is correct for queries that name a category explicitly as the subject
    (e.g. "based on the grading structure" → `specific_only=True`, retrieves GRADING + advisory overlay)
- **Eval framework** (`evals/`) — golden set + full-pipeline harness + router metrics
  - `evals/generate_golden_set.py` — synthesizes 50 stratified questions from live MongoDB chunks
  - `evals/eval_router_metrics.py` — ECE, Intent F1, per-case router accuracy vs golden set
  - `evals/eval_pipeline.py` — end-to-end eval: router → retrieval → generator; captures recall@k, RAGAS, latency
  - `evals/adjudicate_golden_set.py` — LLM adjudicates stratum vs router label disagreements; writes corrected golden set
  - Golden set: 50 questions across all 8 function types (CSCE + ISEN), grounded in live syllabus chunks
  - **Current accuracy on golden set (unadjudicated): 74%** — ~10 failures are golden label errors, not router bugs
  - **Recall@k: 36%** on first run — being investigated (see Known Issues)
  - 75% citation rate on generated responses; 0 pipeline errors
- **Router improvements** (2026-02-23)
  - Added `ADMINISTRATIVE` semantic type for TAMU-tool questions (Canvas, Perusall, Howdy Portal)
    with no specific course_id — routes to `semantic_general` instead of `out_of_scope`
  - `*_combined` functions now always use hybrid retrieval mode regardless of `category_confidence`
    (broad framing benefits from semantic component even when the named category is high-confidence)
  - Suppressed extraction of context-mentioned course IDs (e.g. "I got a B in MATH 151, can I
    take **this** course?" — MATH 151 is background, not the queried course)
  - Added prompt examples for "especially considering" / "including" → `specific_only=false`
- **Generator improvements** (2026-02-25)
  - **Primacy-recency bracketing** in `format_context_xml`: rank 1 → context start, rank 2 → context end,
    ranks 3–N → middle. Combats Lost-in-the-Middle attention degradation.
  - **Single-call comparison architecture** (`generate_comparison`): one Gemini call extracts structured
    JSON via `CourseComparisonTable`; `_render_comparison_markdown()` renders Markdown in Python.
    Saves ~5–10s vs former two-call pattern. Key insight: Gemini in JSON mode fills typed fields
    reliably but not free-form string fields — render Markdown in Python instead.
  - **Streaming output** (`generate_stream`): single-course queries stream tokens via
    `generate_content_stream()`; app.py renders incrementally with `▌` cursor indicator.
  - **Thinking budget reduced**: `THINKING_BUDGET_SEMANTIC` 4096 → 1024, saves ~1–3s per
    hybrid_*/semantic_general call.
  - **Gate 1 citation validation** (`validate_citations_gate1`): regex check for `[Source N]` presence
    in factual responses; result uploaded to Langfuse as `citation_gate1_pass` score.
  - **Gate 2 groundedness scoring**: RAGAS `ResponseGroundedness` metric (Gemini 2.5 Flash-Lite as critic)
    fires asynchronously after each response; score uploaded to Langfuse as `groundedness_score`.
  - **Verbal Uncertainty Calibration**: `build_system_prompt` injects uncertainty language when
    `category_confidence < 0.7`.
  - **Chain-of-Verification** system prompt rule: LLM quotes verbatim in `<thinking>` block before
    paraphrasing, ensuring all claims are grounded.
  - **Model upgrade**: generator switched from `gemini-2.0-flash` → `gemini-2.5-flash`.
  - **Thinking budget**: hybrid_* and semantic_general functions use `thinking_budget=1024`;
    metadata_* functions use `thinking_budget=0` (deterministic extraction).
  - Temperatures: `metadata_*` → 0.0, `hybrid_*` and `semantic_general` → 0.2.
- **Comparison output improvements** (2026-02-25)
  - **4-column table** (Course, Grading, Workload, Prerequisites) replacing the old 6-column table
    (removed Section and Instructor columns which were always empty for difficulty queries).
  - **Detailed Comparison prose section** added after table: subsections Course Overview, Learning
    Outcomes, Topic Complexity, Materials — lets LLM reason about difficulty from retrieved chunks.
  - `CourseComparisonData` schema extended with `course_overview`, `learning_outcomes`,
    `topics_complexity`, `materials` fields; `section` and `instructor` removed.
  - Call 1 `max_output_tokens` raised 2048 → 4096 to fit both courses with all fields.
  - Extraction prompt now explicitly names all courses and instructs LLM not to skip any.
  - `get_missing_sections(course_id)` added to `db/search.py` — derives missing categories from
    `courses.categories_present`; injected into extraction prompt so LLM can distinguish
    "not found in context" from "not found in original syllabus".

### Known Issues
- **`hybrid_combined` requires careful phrasing**: When a category is explicitly named in the query as subject matter, the router correctly sets `specific_only=True` → `hybrid_specific`. Only queries that request a general overview *while also* mentioning a category as background context produce `specific_only=False` → `hybrid_combined`. This is correct behavior, not a bug.
- **Router token budget**: `thinking_budget=512` + `max_output_tokens=1024` — if thinking uses its full budget, only ~512 tokens remain for JSON output. Adequate for current prompt; watch if prompt grows.
- **Langfuse SDK incompatible with Python 3.14**: Official SDK uses `pydantic.v1` which breaks on Python 3.14+. Workaround: custom `MinimalLangfuseClient` in `db/observability.py` posts directly to REST API. Revert to official SDK when they ship a fix.
- **Recall@k is 36%** on the initial golden set run. Primary causes: (1) `metadata_*` path retrieves
  by exact `(course_id, category)` but `source_crn` in golden set is section-specific — a correct answer
  from a different section of the same course counts as MISS; (2) `semantic_general` and `hybrid_default`
  retrieve broad context that may not include the exact source chunk. Recall@k needs per-function
  analysis and potentially a looser match criterion (course_id + category, not CRN-exact).
- **Golden set labels have ~10 stratum errors** — `_derive_router_ground_truth()` in
  `generate_golden_set.py` assigns labels mechanically from strata; synthesized questions that name
  a specific category get wrong labels. Run `evals/adjudicate_golden_set.py` to produce
  `golden_set_v2.jsonl` with LLM-corrected labels. Estimated true router accuracy ~90%.
- **pyOpenSSL + cryptography 46.x deprecation**: System CA certs with negative X.509 serial numbers
  trigger `CryptographyDeprecationWarning` from `pymongo\pyopenssl_context.py`. Currently warnings
  only (not exceptions); will become a hard error in a future cryptography release. Monitor after
  cryptography upgrades.
- **`completeness_check` / `get_missing_sections` unreliable**: `categories_present` in the courses
  collection is derived from whichever chunks were actually ingested, not directly from the parser's
  `completeness_check.missing_sections` field. If a section was parsed but produced a low-quality or
  empty chunk it may still appear in `categories_present`, making `get_missing_sections()` silently
  wrong. Also, the courses collection holds one doc per CRN (section), so `find_one` by `course_id`
  returns an arbitrary section — missing sections may differ across sections of the same course.
  The `missing_note` injected into the extraction prompt may therefore be inaccurate.
  Proper fix: store `completeness_check.missing_sections` directly on the course doc during ingestion.

### Next Steps
1. **Run golden set adjudication** — `python evals/adjudicate_golden_set.py --golden-set tamu_data/logs/golden_set.jsonl --router-results tamu_data/logs/router_metrics.json --output tamu_data/logs/golden_set_v2.jsonl`
2. **Re-run eval on adjudicated set** — measure true router accuracy; also run `--ragas` for faithfulness scores
3. **Investigate recall@k** — redefine hit as `course_id + category` match (not CRN-exact); add per-function breakdown
4. **Expand parsing to all departments** — run `pipeline/process_syllabi.py` without `--department` filter
5. **Add latency percentile tracking** — p50/p95 per function type in Langfuse dashboards
6. **Set score alerts** — Langfuse webhook when faithfulness < 0.5
7. **Observability Phase 2** — prompt management via Langfuse, A/B testing router variables

## Cloud Deployment

- **Service:** Cloud Run (`tamu-bot-service`, `us-central1`)
- **URL:** `https://tamu-bot-service-653181891130.us-central1.run.app`
- **RAG Corpus (legacy):** `projects/glossy-surge-486017-g8/locations/us-south1/ragCorpora/2305843009213693952`
