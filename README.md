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
```

**Tech stack:** Streamlit · MongoDB Atlas · Voyage AI (voyage-3 embeddings + rerank-2) · Gemini 2.5 Flash (router) · Gemini 2.0 Flash (generator) · Scrapy · Pydantic v2

## Quickstart

### Prerequisites

- Python 3.11+
- [MongoDB Atlas](https://www.mongodb.com/atlas) cluster (free M0 works)
- [Voyage AI](https://www.voyageai.com/) API key
- [Google AI](https://aistudio.google.com/) API key (Gemini)

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
│   ├── search.py           # Hybrid search functions
│   ├── router.py           # Query router + orchestrator
│   ├── reranker.py         # Voyage AI reranking
│   └── generator.py        # LLM generation with citations
├── tamu_data/
│   ├── processed/
│   │   └── gemini_parsed/  # 259 structured syllabus JSONs (committed)
│   ├── raw/                # PDFs + scraped JSONL (gitignored, large)
│   └── scraper/            # Scrapy project (catalog + class_search spiders)
└── pipeline/legacy/        # Superseded scripts (Vertex AI, PyMuPDF)
```

## MongoDB Collections

| Collection | Description | Key |
|-----------|-------------|-----|
| `chunks` | Syllabus chunks with 1024-dim Voyage embeddings | `(crn, chunk_index)` |
| `courses` | One doc per section — metadata for aggregations | `crn` |
| `policies` | Deduplicated university boilerplate policies | SHA-256 of policy name |

## Known Issues

- **Comparison query whitespace bug**: Gemini sometimes generates markdown tables with excessive cell padding when chunks are near-duplicates across sections. Fix in progress — deduplicating chunks per `(course_id, category)` before generation.

## License

MIT
