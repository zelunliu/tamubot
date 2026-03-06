# rag/ — RAG Pipeline

## Public API

```python
# Import everything from the package root — never from submodules
from rag import ChunkDoc, CourseDoc, PolicyDoc, VALID_CATEGORIES
from rag import route_retrieve_rerank, classify_query, RouterResult
from rag import generate, generate_stream, generate_comparison
from rag import hybrid_search, search_semantic, search_by_course_categories, get_missing_sections
from rag import fetch_anchor_chunks
from rag import rerank, rerank_multi_course
from rag import get_langfuse, run_ragas_background, compute_ragas_metrics
from rag import compute_dynamic_k, deduplicate_chunks, FUNCTION_CATEGORY_STRATEGIES

reranked, router_result, data_gaps, data_integrity = route_retrieve_rerank(query, trace=None)
# router_result: .function, .course_ids, .specific_categories,
#   .intent_type, .category_confidence, .recurrent_search, .rewritten_query
# data_gaps: [(course_id, category), ...] — pairs with 0 DB results (recurrent only)
# data_integrity: bool — False if any gaps found

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


## Gotchas

- **TAMU gateway `max_tokens`**: min 4096 on ALL `call_llm()` calls — smaller values return empty response
- **`<thinking>` blocks**: system prompt instructs model to write a Chain-of-Verification quote into `<thinking>`. `strip_thinking_blocks()` in `context_builder.py` removes these before returning/streaming to user
- **`generate_eval_search_string`**: uses `max_tokens=4096` (not 256) for TAMU compat
- **Gemini JSON mode**: free-form Markdown fields silently return empty → always render Markdown in Python (`_render_comparison_markdown()`)
- **Primacy-recency** (`format_context_xml`): rank 1 → context start, rank 2 → context end, ranks 3–N → middle
- **Gate 1** (sync, regex): `validate_citations_with_trace()` — checks `[Source N]` presence after generation
- **Gate 2** (async, LLM): `run_groundedness_scoring_background()` — RAGAS groundedness in background thread, score → Langfuse
