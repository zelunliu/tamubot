# TamuBot — Texas A&M Academic Assistant

A RAG-based chatbot that answers questions about TAMU courses, syllabi, grading policies, schedules, and university policies. Built on MongoDB Atlas, Voyage AI, and Gemini.

## Architecture

```
User Query
    │
[Stage 1: Router]  Gemini 2.5 Flash
  8 intent types, multi-course extraction, query rewriting
    │
[Retrieval]  MongoDB Atlas
  Hybrid search (vector + BM25 RRF), k=20 candidates
  Parallel per-course retrieval for comparisons
    │
[Stage 2: Reranker]  Voyage AI rerank-2
  Cross-encoder reranking → top 5
    │
[Stage 3: Generator]  Gemini 2.0 Flash
  XML-tagged context, intent-adaptive prompts, [Source N] citations
    │
Response
    │
[Background] Langfuse + RAGAS
  Trace stored → Faithfulness + AnswerRelevancy scored asynchronously
```

**Tech stack:** Streamlit · MongoDB Atlas · Voyage AI (voyage-3 embeddings + rerank-2) · Gemini 2.5 Flash (router) · Gemini 2.0 Flash (generator) · Langfuse (observability) · RAGAS (evaluation) · Scrapy · Pydantic v2

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

---

## Example Queries

| Query | Intent | What happens |
|-------|--------|-------------|
| What is the grading policy for CSCE 221? | `single_course_lookup` | Retrieves GRADING chunks, cited answer |
| Compare CSCE 120 and CSCE 221 | `multi_course_comparison` | Parallel retrieval, balanced rerank, table |
| Who teaches CSCE 120? | `instructor_query` | INSTRUCTOR category filter |
| How many sections of CSCE 310? | `aggregation_query` | MongoDB `$group` aggregation |
| Can I use ChatGPT in CSCE 221? | `single_course_lookup` + AI_POLICY | Query rewritten with AI synonyms |
| What is the academic integrity policy? | `policy_lookup` | policies collection lookup |
| Howdy! | `out_of_scope` | Canned response, no DB call |

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

Every user query is traced end-to-end in **Langfuse** and asynchronously evaluated by **RAGAS**. See [`OBSERVABILITY.md`](OBSERVABILITY.md) for the full monitoring runbook.

### What gets tracked

| Signal | Tool | Where |
|--------|------|--------|
| Full request trace (Router → Retrieval → Generator) | Langfuse | Traces tab |
| Token usage per stage + thinking tokens | Langfuse | Trace → Generation span |
| Retrieval stats (n_docs, reranker scores) | Langfuse | Trace → Retrieval_Stage span |
| Intent + confidence + rewritten query | Langfuse | Trace → Router_Stage span |
| Faithfulness score | RAGAS → Langfuse | Trace → Scores section |
| Answer Relevancy score | RAGAS → Langfuse | Trace → Scores section |

### Implementation notes

- **Langfuse SDK** (Python) is **not used** — all telemetry posts directly to the Langfuse REST API via `httpx` (`db/observability.py`). This was necessary because the official SDK's Fern-generated layer depends on `pydantic.v1` which breaks on Python 3.14+.
- **RAGAS** runs in a background daemon thread after each response — it does not block the UI.
- **RAGAS embeddings** use Voyage AI (`voyage-3`) to avoid Google Embedding API version compatibility issues.

---

## Project Structure

```
tamubot/
├── app.py                  # Streamlit chat UI
├── config.py               # Environment config
├── process_syllabi.py      # Gemini PDF parser (pipeline step 3)
├── db/
│   ├── models.py           # Pydantic v2 MongoDB models
│   ├── setup_atlas.py      # Create Atlas indexes
│   ├── ingest.py           # Embed + upsert to MongoDB
│   ├── search.py           # Hybrid search (Voyage_Embeddings + MongoDB_Hybrid_Search spans)
│   ├── router.py           # Query router + orchestrator (Router_Stage span)
│   ├── reranker.py         # Voyage AI reranking (Voyage_Reranker span)
│   ├── generator.py        # LLM generation with citations (Generator_Stage generation)
│   └── observability.py    # Langfuse REST client + RAGAS evaluation
├── tamu_data/
│   ├── processed/
│   │   └── gemini_parsed/  # 259 structured syllabus JSONs (committed)
│   ├── raw/                # PDFs + scraped JSONL (gitignored, large)
│   └── scraper/            # Scrapy project (catalog + class_search spiders)
├── pipeline/legacy/        # Superseded scripts (Vertex AI, PyMuPDF)
├── scripts/                # One-off analysis and debug scripts
├── OBSERVABILITY.md        # Langfuse + RAGAS monitoring runbook
└── research_prompts.md     # Gemini Deep Research prompts
```

---

## MongoDB Collections

| Collection | Description | Key |
|-----------|-------------|-----|
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
- 3-stage RAG pipeline: Router (8 intents) → Retrieval + Rerank → Generator (citations)
- **Observability stack** (Phase 1): Langfuse tracing + RAGAS automated evaluation
  - Custom REST-based Langfuse client (Python 3.14 compatible)
  - Full trace hierarchy: Router_Stage → Retrieval_Stage → [Voyage_Embeddings, MongoDB_Hybrid_Search, Voyage_Reranker] → Generator_Stage
  - RAGAS Faithfulness + AnswerRelevancy scored after every response via Voyage AI embeddings

### Known Issues
- **Comparison query whitespace bug**: Gemini generates markdown tables with excessive cell padding when chunks are near-duplicates across sections. Fix: deduplicate by `(course_id, category)` before generation.
- **`general_academic` lacks depth control**: No sub-intent signal — discovery queries and specific-course questions land in the same bucket.

### Next Steps
1. Fix comparison query deduplication
2. Expand parsing to all departments (`python process_syllabi.py` without `--department`)
3. Add latency percentile tracking in Langfuse dashboards
4. Set up Langfuse score alerts (faithfulness < 0.5 threshold)

---

## License

MIT
