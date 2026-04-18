# Langfuse Observability Standardization

**Date**: 2026-04-17  
**Status**: Draft

## Problem

Langfuse observability is spread across ~15 files with 3 integration patterns, causing: flat trace hierarchy, inconsistent naming, dead code (Gate 2), silent eval failures, duplicated critic LLM setup, and no unified config. Each caller (app.py, run_probe.py, run_benchmark.py, eval_chunking.py) has its own ad-hoc trace creation and eval wiring.

## Goals

1. Single `ObservabilityConfig` dataclass controlling tracing + eval per-request
2. Single trace creation function (not 3 different patterns)
3. Standardized eval block system with retry and visible failure scoring
4. Proper trace hierarchy (retrieval sub-spans nest under parent)
5. Consistent dotted-lowercase span naming
6. Explicit cache-hit spans (no silent omissions)
7. Minimize total code

## Architecture

### New Package: `rag/observability/`

```
rag/observability/
    __init__.py          # Public API
    config.py            # ObservabilityConfig + preset factories
    tracing.py           # Singleton + create_trace() + finalize_trace()
    evals.py             # EvalBlock base, registry, runner, critic LLM factory
    ragas_blocks.py      # Concrete eval blocks (faithfulness, relevancy, precision, recall)
```

Replaces `rag/tools/langfuse.py` entirely.

### ObservabilityConfig

```python
@dataclass(slots=True)
class ObservabilityConfig:
    trace_name: str = "tamubot.request"
    tags: list[str] = field(default_factory=list)
    session_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    eval_blocks: list[str] = field(default_factory=list)  # empty = no evals
    eval_async: bool = True       # background thread vs synchronous
    eval_retry: bool = True       # retry once on failure
    enable_generator: bool = True  # False for retrieval-only evals
```

### Preset Factories

| Factory | Caller | Trace Name | Evals | Async |
|---------|--------|------------|-------|-------|
| `prod_config()` | app.py | `tamubot.request` | none | - |
| `probe_config(ragas=True)` | run_probe.py | `tamubot.probe` | faithfulness, answer_relevancy | yes |
| `benchmark_config(ragas=True)` | run_benchmark.py | `tamubot.benchmark` | faithfulness, answer_relevancy | no |
| `chunking_config(ragas=True)` | eval_chunking.py | `tamubot.benchmark` | context_precision, context_recall | no |

### Trace Lifecycle

```python
# Every caller uses this exact pattern:
obs = preset_config(...)
trace, trace_id = create_trace(obs, query=query)
# ... run pipeline with trace= ...
finalize_trace(trace, output=answer)
scores = run_evals(obs, EvalInputs(question=query, contexts=ctx, answer=answer, trace_id=trace_id))
```

`create_trace()` uses `lf.trace()` (proper root trace) instead of `lf.start_observation()`.  
`finalize_trace()` updates output and flushes.

### Eval Block System

```python
class EvalBlock(ABC):
    name: str
    required_fields: tuple[str, ...]
    def compute(self, inputs: EvalInputs) -> dict[str, float]: ...
    def score_failure(self, inputs, error) -> dict[str, float]: ...

@dataclass
class EvalInputs:
    question: str
    contexts: list[str]
    answer: str = ""
    reference: str = ""
    trace_id: str | None = None
```

`run_evals(obs_config, inputs)`:
1. Iterates `obs_config.eval_blocks`
2. Looks up each block in registry
3. Checks required fields are present
4. Runs block with retry (if `eval_retry=True`)
5. On success: posts score to Langfuse
6. On permanent failure: posts score=-1 + failure metadata on trace
7. If `eval_async=True`: all of the above runs in a background daemon thread

### Critic LLM Factory

Single `get_critic_llm()` and `get_critic_embeddings()` singletons, replacing 3 duplicated constructions in current `langfuse.py`.

### Span Naming Convention

| Old | New |
|-----|-----|
| `Router_Stage` | `pipeline.router` |
| `Generator_Stage` | `pipeline.generator` |
| `Generator_Comparison` | `pipeline.generator.comparison` |
| `Recursive_Router_Stage` | `pipeline.router.recursive` |
| `History_Summary` | `pipeline.history.summary` |
| `search.mongo_hybrid` | `pipeline.retrieval.search.hybrid` |
| `search.mongo_semantic` | `pipeline.retrieval.search.semantic` |
| `embed.voyage` | `pipeline.retrieval.embed` |
| `rerank.voyage` | `pipeline.retrieval.rerank` |

Trace names: `tamubot.request`, `tamubot.probe`, `tamubot.benchmark`. Timestamps, experiment names, dataset row IDs go in metadata/tags.

### Cache Hit Visibility

CallbackHandler creates node-level spans before the node function runs. On cache hit, annotate the existing span:

```python
get_client().update_current_observation(metadata={"cache_hit": True})
```

Result: consistent trace shape regardless of cache state. Dashboards can filter by `cache_hit` metadata.

## What Gets Deleted

- `rag/tools/langfuse.py` — replaced by `rag/observability/`
- `score_groundedness()` + `run_groundedness_scoring_background()` (Gate 2, redundant with RAGAS Faithfulness)
- `route_retrieve_rerank()` in `rag/router.py` (trivial wrapper)
- `router_order()` in `rag/router.py` (unused)

## Timing Strategy

- `state["timing_ms"]` stays for benchmark/export convenience (flattened numeric fields for Excel reports)
- Langfuse span durations are the observability source of truth
- No manual duration tracking in Langfuse wrappers — the SDK handles it

## Code Reduction Estimate

| Area | Before (LOC) | After (LOC) |
|------|-------------|-------------|
| `rag/tools/langfuse.py` | 337 | 0 (deleted) |
| `rag/observability/` (new) | 0 | ~200 |
| Caller Langfuse wiring (app + probe + bench + chunking) | ~70 | ~24 |
| Dead code (Gate 2 + wrappers) | ~90 | 0 |
| **Net** | **~497** | **~224** |
