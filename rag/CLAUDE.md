# rag/ — RAG Pipeline Modules

> **Maintenance**: Update this file when public APIs, call signatures, gotchas, or config patterns change.

## Module Dependency Graph

```
app.py
  └── rag/router.py          (route_retrieve_rerank — Stage 1+2 orchestrator)
        ├── rag/llm_client.py (call_llm, stream_llm — unified LLM backend)
        ├── rag/search.py    (hybrid_search, search_semantic, search_by_course_categories)
        ├── rag/reranker.py  (rerank, rerank_multi_course)
        └── rag/generator.py (generate_stream, generate_comparison — Stage 3)
              ├── rag/llm_client.py       (call_llm, stream_llm — unified LLM backend)
              ├── rag/prompts.py          (ROUTER_PROMPT, _FUNCTION_PROMPTS, _FUNCTION_TEMPERATURES)
              ├── rag/context_builder.py  (format_context_xml)
              ├── rag/gates.py            (validate_citations_gate1, validate_citations_with_trace)
              ├── rag/comparison_schemas.py (CourseComparisonTable Pydantic schema)
              └── rag/observability.py    (run_groundedness_scoring_background — Gate 2)
```

## Public API Entry Points

```python
# Full pipeline (app.py calls these)
from rag.router import route_retrieve_rerank   # returns (reranked_results, RouterResult)
from rag.generator import generate_stream      # yields text chunks (single-course)
from rag.generator import generate_comparison  # returns Markdown string (multi-course)
```

## Config Access Pattern

**Always import from `config.py`**. Never call `os.getenv()` directly in rag/ modules.

```python
import config
config.GENERATION_MODEL         # Gemini model name (direct path)
config.TAMU_MODEL               # protected.gemini-2.5-flash (TAMU gateway)
config.USE_TAMU_API             # True when TAMU_API_KEY is set
config.THINKING_BUDGET_SEMANTIC # 1024
config.get_genai_client()       # lazy singleton google.genai.Client
config.get_tamu_client()        # lazy singleton openai.OpenAI (TAMU gateway)
```

## TAMU AI API / LLM Client

All LLM calls in `router.py` and `generator.py` go through `rag/llm_client.py`:

```python
from rag.llm_client import call_llm, stream_llm, LLMResult

result = call_llm(messages, temperature=0, max_tokens=4096, json_mode=True, thinking_budget=512)
text = result.text
# result.input_tokens / output_tokens / thinking_tokens — None on TAMU path

for token in stream_llm(messages, temperature=0.2, max_tokens=4096, thinking_budget=1024):
    ...
```

`llm_client.py` selects the backend automatically via `config.USE_TAMU_API`:
- **TAMU path**: OpenAI-compat gateway, always `stream=True` (SSE quirk), no token counts
- **Gemini path**: google-genai SDK, token counts returned in `LLMResult`

Do **not** call `config.get_tamu_client()` or `config.get_genai_client()` directly in
`router.py` or `generator.py` — always go through `llm_client`.

Langfuse spans: use `config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL`
for the model name.  Token counts from `LLMResult` are None on TAMU path — check before logging.

## Singleton Clients

`search.py` uses `_get_db()` and `_get_voyage()` module-level singletons.
Do not instantiate `MongoClient` or `voyageai.Client` directly in new code.

## 8-Function Decision Matrix

Derived by pure Python in `_derive_function()` — no LLM judgment:

```
course_ids  semantic_intent  specific_categories  specific_only  → function           retrieval_mode
empty       true             any                  any            → semantic_general    semantic
empty       false            any                  any            → out_of_scope        —
present     false            empty                —              → metadata_default    metadata
present     false            populated            true           → metadata_specific   metadata
present     false            populated            false          → metadata_combined   hybrid
present     true             empty                —              → hybrid_default      hybrid
present     true             populated            true           → hybrid_specific     hybrid
present     true             populated            false          → hybrid_combined     hybrid
```

Note: `*_combined` always uses hybrid retrieval regardless of `category_confidence`.

## Thinking Budgets

- `THINKING_BUDGET_METADATA = 0` — metadata_* functions (deterministic extraction)
- `THINKING_BUDGET_SEMANTIC = 1024` — hybrid_* and semantic_general (complex reasoning)

## Per-Function Temperatures

- `metadata_*` → 0.0  (deterministic, maximum context fidelity)
- `hybrid_*` → 0.2    (advisory reasoning, linguistic synthesis)
- `semantic_general` → 0.2
- `out_of_scope` → 0.0

## Critical Gotchas

### Gemini JSON Mode
When using `response_mime_type="application/json"` + `response_schema`, Gemini reliably fills
typed fields (`str`, `list`) but does **NOT** reliably populate a free-form Markdown `str` field.
→ Always render Markdown in Python from structured data (`_render_comparison_markdown()`).

### Primacy-Recency Bracketing (`format_context_xml`)
Reranked results are reordered before feeding to Gemini to combat Lost-in-the-Middle degradation:
- Rank 1 → context **start** (primacy position)
- Rank 2 → context **end** (recency/nearest-to-query position)
- Ranks 3–N → middle (descending order)

### Gates
- **Gate 1** (regex, synchronous): `validate_citations_with_trace()` fires immediately after
  generation; checks for `[Source N]` presence in factual responses.
- **Gate 2** (LLM, asynchronous): `run_groundedness_scoring_background()` in `observability.py`
  fires in a background thread via RAGAS `ResponseGroundedness`; score uploaded to Langfuse.

## Key Files Quick Reference

| File | Responsibility |
|------|---------------|
| `router.py` | Stage 1: LLM variable extraction + pure-Python routing + retrieval orchestration |
| `llm_client.py` | Unified LLM backend: `call_llm()` (blocking) + `stream_llm()` (streaming) |
| `search.py` | MongoDB Atlas vector + hybrid + metadata search |
| `reranker.py` | Voyage AI rerank-2 cross-encoder reranking |
| `generator.py` | Stage 3: single-course `generate_stream()`, multi-course `generate_comparison()` |
| `prompts.py` | All prompt strings: router, function prompts, semantic overlays, temperatures |
| `context_builder.py` | `format_context_xml()` — primacy-recency XML context assembly |
| `gates.py` | Gate 1 citation validation (regex + Langfuse scoring) |
| `models.py` | Pydantic v2 models: ChunkDoc, PolicyDoc, CourseDoc |
| `comparison_schemas.py` | CourseComparisonTable schema for single-call extraction |
| `observability.py` | Langfuse tracing + RAGAS eval + Gate 2 groundedness scoring |
