# rag/ — RAG Pipeline

## Public API

```python
# Import everything from the package root — never from submodules
from rag import route_retrieve_rerank, classify_query, RouterResult
from rag import generate, generate_stream, generate_comparison
from rag import hybrid_search, search_semantic, search_by_course_categories, get_missing_sections
from rag import rerank, rerank_multi_course
from rag import get_langfuse, run_ragas_background, compute_ragas_metrics
from rag import compute_dynamic_k, deduplicate_chunks, FUNCTION_CATEGORY_STRATEGIES

reranked, router_result = route_retrieve_rerank(query, trace=None)
# router_result: .function, .course_ids, .specific_categories,
#   .semantic_intent, .semantic_type, .category_confidence, .recurrent_search, .rewritten_query

# Observability
lf = get_langfuse()                           # MinimalLangfuseClient or None
trace = lf.trace(name, input, metadata)
lf.flush()                                    # call after each query in scripts
# Trace URL: f"{config.LANGFUSE_BASE_URL}/trace/{trace.id}"
```

## LLM Client

All LLM calls go through `rag/llm_client.py` — do NOT call `config.get_tamu_client()` / `config.get_genai_client()` directly in router/generator:

```python
result = call_llm(messages, temperature=0, max_tokens=4096, json_mode=True, thinking_budget=512)
for token in stream_llm(messages, temperature=0.2, max_tokens=4096, thinking_budget=1024):
```

- `result.input_tokens / output_tokens` — None on TAMU path (SSE, no token counts exposed)
- Langfuse model name: `config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL`

## 8-Function Routing Matrix

Pure Python in `_derive_function()` — no LLM judgment:

```
course_ids  recurrent_search  semantic_intent  specific_categories  specific_only  → function            mode
empty       any               true             any                  any            → semantic_general     semantic
empty       any               false            any                  any            → out_of_scope         —
present     true              any              empty                —              → recurrent_default    hybrid (2-stage)
present     true              any              populated            true           → recurrent_specific   hybrid (2-stage)
present     true              any              populated            false          → recurrent_combined   hybrid (2-stage)
present     false             any              empty                —              → metadata_default     metadata
present     false             any              populated            true           → metadata_specific    metadata
present     false             any              populated            false          → metadata_combined    metadata
```

`recurrent_*` = two-stage: (1) metadata fetch anchor course → (2) corpus-wide hybrid discovery.
Thinking budget: `metadata_*` = 0, `recurrent_*`/`semantic_general` = 1024. Temps: `metadata_*` = 0.0, others = 0.2.

## Gotchas

- **Gemini JSON mode**: free-form Markdown fields silently return empty → always render Markdown in Python (`_render_comparison_markdown()`)
- **Primacy-recency** (`format_context_xml`): rank 1 → context start, rank 2 → context end, ranks 3–N → middle
- **Gate 1** (sync, regex): `validate_citations_with_trace()` — checks `[Source N]` presence after generation
- **Gate 2** (async, LLM): `run_groundedness_scoring_background()` — RAGAS groundedness in background thread, score → Langfuse
