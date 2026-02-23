# TamuBot — Texas A&M Academic Assistant

A RAG-based chatbot that answers questions about TAMU courses, syllabi, grading policies, schedules, and university policies. Built on MongoDB Atlas, Voyage AI, and Gemini.

**Tech stack:** Streamlit · MongoDB Atlas · Voyage AI (voyage-3 embeddings + rerank-2) · Gemini 2.5 Flash (router) · Gemini 2.0 Flash (generator) · Langfuse (observability) · RAGAS (evaluation) · Scrapy · Pydantic v2

---

## High-Level System Flow

```mermaid
%%{init: {'theme': 'dark', 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 70}}}%%
flowchart LR
    A([Student]) -->|Question| App[Streamlit\nWeb App]

    subgraph S1["Stage 1 — Router · Gemini 2.5 Flash"]
        direction TB
        Router["Gemini 2.5 Flash<br/>Variable Extraction<br/>course_ids · categories<br/>semantic_intent"]
        FD["Function Derivation<br/>Pure Python"]
        Router --> FD
    end

    subgraph S2["Stage 2 — Retrieval · MongoDB + Voyage AI"]
        direction TB
        Orch[Retrieval\nOrchestrator]
        MetaDB[(MongoDB Atlas\nIndex Lookup)]
        VoyEmbed[Voyage voyage-3\nEmbeddings]
        Search[(vectorSearch\n+ fulltext RRF)]
        Reranker[Voyage rerank-2\nCross-encoder]
        Orch --> MetaDB
        Orch --> VoyEmbed --> Search --> Reranker
    end

    subgraph S3["Stage 3 — Generator · Gemini 2.0 Flash"]
        direction TB
        Dedup[Deduplication\nMiddleware]
        Gen["Gemini 2.0 Flash<br/>Adaptive Prompt + Citations"]
        Dedup --> Gen
    end

    Canned[Canned\nResponse]
    Obs["Langfuse Tracing<br/>RAGAS Eval"]

    App --> S1
    FD -->|out_of_scope| Canned --> App
    FD --> S2
    MetaDB --> Dedup
    Reranker --> Dedup
    Gen -->|"Answer + [Source N]"| App --> A
    App -.->|async| Obs

    classDef user     fill:#004D40,stroke:#4DB6AC,color:#B2DFDB
    classDef router   fill:#880E4F,stroke:#F06292,color:#F8BBD0
    classDef derive   fill:#4A148C,stroke:#BA68C8,color:#E1BEE7
    classDef canned   fill:#212121,stroke:#757575,color:#BDBDBD
    classDef orchestr fill:#3E2723,stroke:#A1887F,color:#D7CCC8
    classDef mongo    fill:#1A237E,stroke:#7986CB,color:#C5CAE9
    classDef voyage   fill:#01579B,stroke:#4FC3F7,color:#E1F5FE
    classDef obs      fill:#1B5E20,stroke:#66BB6A,color:#C8E6C9

    class A,App user
    class Router,Gen router
    class FD derive
    class Canned canned
    class Orch,Dedup orchestr
    class MetaDB,Search mongo
    class VoyEmbed,Reranker voyage
    class Obs obs
```

---

## RAG Pipeline Detail

```mermaid
%%{init: {'theme': 'dark', 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 60}}}%%
flowchart LR
    Q([User Question]) --> INVOKE{Route &\nRetrieve}

    subgraph ROUTER["Router — Gemini 2.5 Flash · temp=0 · thinking_budget=512"]
        direction TB
        LLM["Gemini 2.5 Flash<br/>JSON · max_output_tokens=1024"]
        VARS["course_ids · specific_categories<br/>category_confidence · specific_only<br/>semantic_intent · semantic_type<br/>rewritten_query"]
        FN{"function\n+ retrieval_mode"}
        LLM --> VARS --> FN
    end

    subgraph RETRIEVAL["Retrieval — MongoDB Atlas + Voyage AI voyage-3"]
        direction TB
        META["search_by_course_categories<br/>No embedding · exact index"]
        EMB[Voyage voyage-3\nQuery Embedding]
        HYB["hybrid_search · per course<br/>RRF vectorSearch + fulltext"]
        SEM["search_semantic<br/>full corpus vectorSearch"]
        CHUNK[(MongoDB Atlas\nchunks collection)]
        RRK[Voyage rerank-2\nCross-encoder]
        META --> CHUNK
        EMB --> HYB --> CHUNK
        EMB --> SEM --> CHUNK
        CHUNK --> RRK
    end

    subgraph DEDUP_SUB["Deduplication Middleware"]
        DEDUP["_deduplicate_chunks<br/>best chunk per (course_id, category)"]
    end

    subgraph GENERATOR["Generator — Gemini 2.0 Flash"]
        direction TB
        CTX["XML Context Builder<br/>Source N tags · recency ordered"]
        PROMPT["System Prompt Assembly<br/>function-adaptive + semantic_type overlay"]
        GEN["Gemini 2.0 Flash<br/>temp: metadata→0.0 · hybrid→0.1"]
        ANS[Response + Citations]
        CTX --> PROMPT --> GEN --> ANS
    end

    OOS[out_of_scope\nCanned Response]
    REPLY([Reply to Student])
    TRACE["Langfuse · 5-span trace<br/>RAGAS Faithfulness + AnswerRelevancy"]

    INVOKE -->|structured extraction| ROUTER
    FN -->|"∅ ids · sem=false"| OOS --> REPLY
    FN -->|"catconf ≥ 0.7"| META
    FN -->|hybrid| EMB
    FN -->|semantic| EMB
    META --> DEDUP
    RRK --> DEDUP
    DEDUP --> CTX
    ANS --> REPLY
    ANS -.->|async| TRACE

    classDef input    fill:#004D40,stroke:#4DB6AC,color:#B2DFDB
    classDef router   fill:#880E4F,stroke:#F06292,color:#F8BBD0
    classDef vars     fill:#3E2723,stroke:#A1887F,color:#D7CCC8
    classDef derive   fill:#4A148C,stroke:#BA68C8,color:#E1BEE7
    classDef canned   fill:#212121,stroke:#757575,color:#BDBDBD
    classDef voyage   fill:#01579B,stroke:#4FC3F7,color:#E1F5FE
    classDef mongo    fill:#1A237E,stroke:#7986CB,color:#C5CAE9
    classDef obs      fill:#1B5E20,stroke:#66BB6A,color:#C8E6C9

    class Q,REPLY input
    class LLM,GEN,ANS router
    class VARS,CTX,PROMPT,DEDUP vars
    class FN,INVOKE derive
    class OOS canned
    class EMB,HYB,SEM,RRK voyage
    class META,CHUNK mongo
    class TRACE obs
```

---

## Router — Variable Extraction Schema

The router is a single Gemini 2.5 Flash call that **extracts structured facts** from the query. There is no intent classification step — the retrieval function is derived mechanically from the extracted variables in pure Python.

### Variables Extracted by the LLM

| Variable | Type | Description |
|---|---|---|
| `course_ids` | `list[str]` | Normalized course IDs found in query (e.g. `["CSCE 638", "CSCE 670"]`) |
| `specific_categories` | `list[str]` | Syllabus categories being asked about (e.g. `["GRADING", "AI_POLICY"]`) |
| `category_confidence` | `float 0–1` | Confidence in the category extraction |
| `specific_only` | `bool` | `True` → the query asks *only* about those categories (not a broad overview) |
| `semantic_intent` | `bool` | `True` → question has advisory/evaluative/opinion component |
| `semantic_type` | `str \| None` | `ACADEMIC · CAREER · DIFFICULTY · PLANNING · GENERAL` |
| `rewritten_query` | `str` | Expanded for retrieval (synonyms, slang expansion) |
| `section` | `str \| None` | Section number if mentioned |

### Function Derivation Matrix (pure Python)

| `course_ids` | `semantic_intent` | `specific_categories` | `specific_only` | **function** |
|---|---|---|---|---|
| empty | `True` | any | any | `semantic_general` |
| empty | `False` | any | any | `out_of_scope` |
| present | `False` | empty | — | `metadata_default` |
| present | `False` | populated | `True` | `metadata_specific` |
| present | `False` | populated | `False` | `metadata_combined` |
| present | `True` | empty | — | `hybrid_default` |
| present | `True` | populated | `True` | `hybrid_specific` |
| present | `True` | populated | `False` | `hybrid_combined` |

### Retrieval Mode Derivation (pure Python)

```
retrieval_mode = "semantic"  if course_ids is empty
               = "metadata"  if category_confidence >= 0.7
               = "hybrid"    otherwise
```

**Metadata path** → `search_by_course_categories()` — exact index lookup, no embedding, no reranking.
**Hybrid path** → `hybrid_search()` — RRF of `$vectorSearch` + `$search`, Voyage voyage-3, then rerank-2.
**Semantic path** → `search_semantic()` — `$vectorSearch` over full corpus, then rerank-2.

**Multi-course** (`len(course_ids) > 1`): all functions run parallel per-course fetch, then `rerank_multi_course()` with round-robin interleaving.

### Retrieval Config Per Function

| Function | retrieve_k | rerank_k | Notes |
|---|---|---|---|
| `metadata_default` | 10 | 0 | No reranking (exact lookup) |
| `metadata_specific` | 10 | 0 | No reranking (exact lookup) |
| `metadata_combined` | 10 | 0 | No reranking (exact lookup) |
| `semantic_general` | 30 | 10 | Full corpus search |
| `hybrid_default` | 12 | 3 | Per course |
| `hybrid_specific` | 10 | 3 | Per course |
| `hybrid_combined` | 15 | 4 | Per course |

### Default Summary Categories

When `specific_categories` is empty (or for the `*_combined` base), the system fetches:

```python
DEFAULT_SUMMARY_CATEGORIES = ["COURSE_OVERVIEW", "PREREQUISITES", "LEARNING_OUTCOMES"]
```

### Query Rewriting Rules

| Slang / shorthand | Expanded to |
|---|---|
| "late work" | "attendance makeup deadline extensions late submission" |
| "ChatGPT", "AI tools" | "AI policy artificial intelligence generative AI tools" |
| "prereqs" | "prerequisites required courses corequisites" |
| "prof", "teacher" | "instructor professor" |
| "grade breakdown" | "grading policy grade distribution weight percentage" |

Course IDs are normalized: `csce638` → `CSCE 638`, `CSCE-670` → `CSCE 670`.

---

## Generator Behavior

Each function gets a tailored system prompt and temperature. An advisory overlay from `semantic_type` is appended for `hybrid_*` and `semantic_general` functions.

### Function Prompts

| Function | Temp | Framing |
|---|---|---|
| `metadata_default` | 0.0 | General course overview — covers overview, prerequisites, learning outcomes |
| `metadata_specific` | 0.0 | Focused on requested categories — precise and complete |
| `metadata_combined` | 0.0 | Specific categories in context of full overview |
| `semantic_general` | 0.0 | Broad question — search-based answer, flags insufficient evidence |
| `hybrid_default` | 0.1 | Advisory question — grounds all opinions in course-context facts |
| `hybrid_specific` | 0.1 | Advisory + specific category — uses category evidence for opinion |
| `hybrid_combined` | 0.1 | Advisory + overview — covers all categories, advisory synthesis |

### Semantic Type Advisory Overlays

Appended to the system prompt for `hybrid_*` and `semantic_general` functions:

| `semantic_type` | Advisory instruction |
|---|---|
| `ACADEMIC` | Discuss learning outcomes, topics covered, academic content |
| `CAREER` | Discuss how course content relates to industry applications and career paths |
| `DIFFICULTY` | Use grading weights, prerequisites, attendance requirements as evidence of rigor |
| `PLANNING` | Help the student understand course fit in their academic progression |
| `GENERAL` | Address the advisory aspect using evidence from the course context |

**Multi-course overlay:** when `len(course_ids) > 1`, a comparison instruction is automatically appended (Markdown table, no alignment padding, `[Source N]` citations per cell).

**Recency bias:** results are reversed before context formatting so the highest-ranked chunk sits directly above the user question.

**Verification rule (global):** the model must identify which chunk contains the answer before responding. If none do, it states "I cannot find that information in the provided context" and does not use training data.

---

## Router Evaluation Results

**Dry-run evaluation** — router only, 34 test cases, CSCE 638 + CSCE 670 as default test courses (2026-02-23).

**Function accuracy: 34/34 (100%)** after two prompt fixes:
1. Increased `max_output_tokens` from 512 → 1024 (JSON was truncating mid-response when thinking budget consumed tokens)
2. Tightened `semantic_intent` rules to require TAMU-academic scope and distinguish factual comparisons from evaluative queries

### Results by Function

| Function | Cases | Accuracy | Typical catconf | Retrieval mode |
|---|---|---|---|---|
| `metadata_specific` | 10 | 10/10 | 0.95 | metadata |
| `metadata_default` | 3 | 3/3 | 1.00 | metadata |
| `metadata_combined` | 2 | 2/2 | 0.85–0.90 | metadata |
| `hybrid_specific` | 4 | 4/4 | 0.95 | metadata |
| `hybrid_default` | 6 | 6/6 | 0.0–1.0 | hybrid/metadata |
| `semantic_general` | 5 | 5/5 | 0.0–0.95 | semantic |
| `out_of_scope` | 4 | 4/4 | 0.00 | — |

### Notable Boundary Behaviors

| Query type | Behavior | Explanation |
|---|---|---|
| `"Is CSCE 638 strict about its AI policy?"` | `hybrid_specific` · `semantic_type=ACADEMIC` | Evaluative word ("strict") triggers `semantic_intent=True`; explicit category → `specific_only=True` |
| `"Compare the AI policies of CSCE 638 and CSCE 670"` | `metadata_specific` (multi-course) | Factual comparison → `semantic_intent=False`; parallel per-course fetch |
| `"Is CSCE 638 harder than CSCE 670?"` | `hybrid_default` · `semantic_type=DIFFICULTY` | Opinion → `semantic_intent=True`; no specific category → `hybrid_default` |
| `"Does CSCE 638 have heavy workload based on the grading structure?"` | `hybrid_specific` · `semantic_type=DIFFICULTY` | Category explicitly named ("based on grading structure") → `specific_only=True` → `hybrid_specific`, not `hybrid_combined` |
| `"What is the TAMU academic integrity policy?"` | `semantic_general` · `semantic_type=ACADEMIC` | No course_id; TAMU-academic discovery → `semantic_intent=True` |
| `"What are the best restaurants near TAMU?"` | `out_of_scope` | Non-TAMU topic → `semantic_intent=False` → `out_of_scope` |

---

## Example Queries

| Query | Function | What happens |
|---|---|---|
| `"What is the grading breakdown for CSCE 638?"` | `metadata_specific` | GRADING chunk fetched by index, cited answer, temp=0.0 |
| `"Tell me about CSCE 670"` | `metadata_default` | COURSE_OVERVIEW + PREREQUISITES + LEARNING_OUTCOMES fetched |
| `"Compare the AI policies of CSCE 638 and CSCE 670"` | `metadata_specific` (×2) | Parallel per-course index lookup, comparison table generated |
| `"Is CSCE 638 strict about its AI policy?"` | `hybrid_specific` | AI_POLICY chunk + ACADEMIC advisory overlay |
| `"Is CSCE 638 harder than CSCE 670?"` | `hybrid_default` | Parallel hybrid search, DIFFICULTY overlay, opinion synthesis |
| `"Which courses will help me become an AI engineer?"` | `semantic_general` | Full-corpus vector search, CAREER overlay |
| `"What is the TAMU academic integrity policy?"` | `semantic_general` | Full-corpus search surfaces UNIVERSITY_POLICIES chunk |
| `"Howdy!"` | `out_of_scope` | Canned response, no DB call |

---

## Quickstart

### Prerequisites

- Python 3.11+ (tested on 3.14)
- [MongoDB Atlas](https://www.mongodb.com/atlas) cluster (free M0 works)
- [Voyage AI](https://www.voyageai.com/) API key
- [Google AI](https://aistudio.google.com/) API key (Gemini)
- [Langfuse](https://cloud.langfuse.com) account (free tier works)

### Setup

```bash
git clone https://github.com/artemkorolev1/tamubot
cd tamubot

python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

cp .env.example .env
# Edit .env and fill in your API keys
```

### Configure `.env`

```env
MONGODB_URI=mongodb+srv://...
MONGODB_DB=tamubot

VOYAGE_API_KEY=...
GOOGLE_API_KEY=...

RETRIEVAL_BACKEND=mongodb

# Observability (optional but recommended)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### Load data into MongoDB

The parsed syllabus JSONs are included in `tamu_data/processed/gemini_parsed/` (259 CSCE + ISEN courses, Spring 2026).

```bash
# Create Atlas indexes (vector + text + metadata)
python -m db.setup_atlas

# Embed and ingest all parsed JSONs
python -m db.ingest

# Or ingest a single department
python -m db.ingest --department CSCE
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

### Evaluate

```bash
# Router dry-run (no MongoDB needed)
python scripts/eval_pipeline.py --dry-run

# Full pipeline (requires MongoDB + all API keys)
python scripts/eval_pipeline.py

# Single function type
python scripts/eval_pipeline.py --function metadata_specific

# Single test
python scripts/eval_pipeline.py --test-id 13
```

---

## Data Pipeline

To scrape fresh data and rebuild from scratch:

```bash
# 1. Scrape academic catalog
make scrape-catalog

# 2. Scrape course sections + download syllabi PDFs
make scrape-classes

# 3. Parse PDFs with Gemini (resumes automatically)
python process_syllabi.py
# For a single department: python process_syllabi.py --department CSCE
# Retry failed files:      python process_syllabi.py --retry-errors

# 4. Rebuild MongoDB
python -m db.setup_atlas
python -m db.ingest
```

---

## Observability & Evaluation

Every user query is traced end-to-end in **Langfuse** and asynchronously evaluated by **RAGAS**.

### What gets tracked

| Signal | Tool | Where |
|---|---|---|
| Full request trace (Router → Retrieval → Generator) | Langfuse | Traces tab |
| Token usage per stage + thinking tokens | Langfuse | Trace → Generation span |
| Retrieval stats (n_docs, dedup count) | Langfuse | Trace → Retrieval_Stage span |
| function + retrieval_mode + category_confidence | Langfuse | Trace → Router_Stage span |
| Faithfulness score | RAGAS → Langfuse | Trace → Scores section |
| Answer Relevancy score | RAGAS → Langfuse | Trace → Scores section |

### Implementation notes

- **Langfuse SDK** is **not used** — all telemetry posts directly to the Langfuse REST API via `httpx` (`db/observability.py`). Required because the official SDK depends on `pydantic.v1` which breaks on Python 3.14+.
- **RAGAS** runs in a background daemon thread after each response — it does not block the UI.
- **RAGAS embeddings** use Voyage AI (`voyage-3`) to avoid Google Embedding API compatibility issues.

---

## Project Structure

```
tamubot/
├── app.py                  # Streamlit chat UI
├── config.py               # Env config + FUNCTION_RETRIEVAL_CONFIG + DEFAULT_SUMMARY_CATEGORIES
├── process_syllabi.py      # Gemini PDF parser (pipeline step 3)
├── db/
│   ├── models.py           # Pydantic v2 MongoDB models
│   ├── setup_atlas.py      # Create Atlas indexes (vector + text + compound metadata)
│   ├── ingest.py           # Embed + upsert to MongoDB (Voyage voyage-3)
│   ├── search.py           # hybrid_search · search_semantic · search_by_course_categories
│   ├── router.py           # Variable extraction + function derivation + orchestrator
│   ├── reranker.py         # Voyage rerank-2 (single + multi-course)
│   ├── generator.py        # Function-adaptive prompts + semantic_type overlays + citations
│   └── observability.py    # Langfuse REST client + RAGAS evaluation
├── tamu_data/
│   ├── processed/
│   │   └── gemini_parsed/  # 259 structured syllabus JSONs (committed)
│   ├── raw/                # PDFs + scraped JSONL (gitignored, large)
│   └── scraper/            # Scrapy project (catalog + class_search spiders)
├── scripts/
│   └── eval_pipeline.py    # 34-case evaluation harness (router dry-run + full pipeline)
├── pipeline/legacy/        # Superseded scripts (Vertex AI, PyMuPDF)
├── OBSERVABILITY.md        # Langfuse + RAGAS monitoring runbook
└── research_prompts.md     # Gemini Deep Research prompts
```

---

## MongoDB Collections

| Collection | Description | Key |
|---|---|---|
| `chunks` | Syllabus chunks with 1024-dim Voyage embeddings | `(crn, chunk_index)` |
| `courses` | One doc per section — metadata for aggregations | `crn` |
| `policies` | Deduplicated university boilerplate policies | SHA-256 of policy name |

---

## Current Status (as of 2026-02-23)

### Completed

- Scrapy spiders for catalog + class search (all departments)
- Syllabus PDF download — 7,970 PDFs across all departments
- Gemini PDF parsing — 259/259 CSCE + ISEN files parsed
- MongoDB Atlas integration: models, indexes, ingestion, hybrid search
- **3-stage RAG pipeline** (Router → Retrieval+Rerank → Generator) with XML context and `[Source N]` citations
- **New router schema** — structured variable extraction replaces intent classification; function derived mechanically from variables; 34-case dry-run eval at 100% function accuracy
- **`search_by_course_categories()`** — no-embedding exact index lookup for high-confidence metadata paths
- **Observability stack** (Phase 1): Langfuse tracing + RAGAS automated evaluation

### Known Issues

- **`hybrid_combined` requires very specific phrasing**: When a category is explicitly named in a query ("based on the grading structure"), the router correctly extracts `specific_only=True` → `hybrid_specific`. Only queries that request a broad overview *and* mention a category as background context produce `specific_only=False` → `hybrid_combined`.
- **Langfuse SDK incompatible with Python 3.14**: Workaround in `db/observability.py` (direct REST). Revert to official SDK when fixed upstream.
- **Router token budget**: With `thinking_budget=512` and `max_output_tokens=1024`, the router uses ~600–900 total output tokens per call. Long rewritten queries in `hybrid_combined` cases may approach the limit.

### Next Steps

1. Expand parsing to all departments (`python process_syllabi.py` without `--department`)
2. Run full pipeline eval with ingested MongoDB (retrieval quality + citation rate)
3. Add latency percentile tracking (p50/p95 per function) in Langfuse dashboards
4. Set up Langfuse score alerts (faithfulness < 0.5)
5. Observability Phase 2 — prompt management via Langfuse, A/B testing router variables

---

## License

MIT
