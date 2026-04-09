# Chunking Strategy Evaluation — Design Spec

**Date:** 2026-04-09  
**Status:** Approved

## Context

Existing benchmarks measure end-to-end pipeline performance (router → retrieval → generation). This spec adds a dedicated retrieval-only benchmark to compare chunking strategies by their retrieval quality, without running the generator. The primary output is per-query and run-level metrics logged to Langfuse for side-by-side experiment comparison in the UI.

## Goals

- Measure retrieval quality (not answer quality) across chunking configurations
- Compute precision, recall, F1, hit rate, and token efficiency per query
- Log everything to Langfuse (per-query traces + run-level aggregates) for clean UI comparison
- Keep RAGAS calls optional (they cost LLM tokens; fast runs use embedding-only metrics)

## Files Changed

| File | Change |
|------|--------|
| `evals/eval_chunking.py` | New script — main eval logic |
| `rag/tools/langfuse.py` | Add `compute_retrieval_ragas()` function |
| `Makefile` | Add `eval-chunking` target |

## Metrics

### Embedding-based (always computed, cheap)
- **`precision_at_k`**: `n_relevant_in_topk / k` — Voyage-3 cosine similarity ≥ threshold labels each chunk
- **`hit_rate_at_k`**: `1.0` if any chunk is relevant, else `0.0`
- **`retrieved_tokens`**: `sum(len(chunk["content"]) // 4)` across reranked chunks (same approximation as `chunker_v3.py`)

### RAGAS-based (optional, `--ragas` flag, costs LLM tokens)
- **`context_recall`** / **`recall_at_k`**: RAGAS `ContextRecall` — fraction of `reference_answer` claims covered by retrieved chunks
- **`context_precision`**: RAGAS `ContextPrecision` — fraction of retrieved chunks that are relevant (LLM-judged)
- **`f1_at_k`**: `2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)`

Without `--ragas`, only the three embedding-based metrics are computed.

## Variables (fixed per run, configurable via CLI)

| Variable | Default | CLI flag |
|----------|---------|----------|
| `chunk_size_tokens` | from ingestion config | informational only (not re-ingested here) |
| `chunk_overlap_tokens` | from ingestion config | informational only |
| `similarity_threshold` | `0.85` | `--threshold` |
| `top_k` (rerank cutoff) | from `compute_dynamic_k()` | `--top-k` |

## `evals/eval_chunking.py`

### CLI

```bash
python evals/eval_chunking.py \
  --golden-set tamu_data/evals/golden_sets/golden_20260313_draft_v1.jsonl \
  [--experiment chunk_600_ov100] \
  [--top-k 7] \
  [--threshold 0.85] \
  [--ragas] \
  [--output tamu_data/evals/reports/chunking_YYYYMMDD.json]
```

`--golden-set` is required (no default). Use the 10-question sample for smoke tests, full 50-question set for real benchmarks.

### Key functions

1. **`load_golden_set(path)`** — reads JSONL; warns (does not skip) on missing `reference_answer` when `--ragas` is off; skips those items when `--ragas` is on (recall requires a reference)
2. **`retrieve_for_query(query, rr, retrieve_k, rerank_k)`** — calls `hybrid_search()` or `search_semantic()` based on router function, then `rerank()`; all imported from `rag` public API
3. **`compute_embedding_metrics(query, chunks, threshold)`** — calls `label_relevant()` from `evals/eval_retrieval_metrics.py`, returns `precision_at_k`, `hit_rate_at_k`, `retrieved_tokens`
4. **`run_eval(...)`** — main loop: routes each query, retrieves, computes metrics, logs to Langfuse
5. **`log_to_langfuse(lf, trace, dataset_item, metrics, run_name)`** — scores each metric on the trace, links trace to dataset item
6. **`print_summary(results, run_name)`** — aligned console table of per-query and aggregate metrics

### Retrieval path

Uses the same imports as `eval_retrieval_metrics.py`:
```python
from rag import classify_query, compute_dynamic_k, hybrid_search, rerank, search_semantic
from evals.eval_retrieval_metrics import label_relevant
```

Router is called to get `course_ids` and `function` for filter construction. `--top-k` overrides `rerank_k` from `compute_dynamic_k()`.

## `rag/tools/langfuse.py` — new function

```python
def compute_retrieval_ragas(
    question: str,
    contexts: list[str],
    reference: str,
    trace_id: Optional[str] = None,
) -> dict:
```

- `SingleTurnSample(user_input=question, retrieved_contexts=contexts, reference=reference)`
- Metrics: `ContextPrecision(llm=critic_llm)` + `ContextRecall(llm=critic_llm)`
- Same TAMU LLM setup as `compute_ragas_metrics()` (copy pattern exactly)
- Same litellm budget cap disable
- Uploads scores to trace via `lf.create_score()` if `trace_id` provided
- Returns `dict[str, float]` — keys: `context_precision`, `context_recall`

## Langfuse Integration

| Level | What gets logged |
|-------|-----------------|
| Per-query | `lf.trace()` → one score per metric → linked to dataset item via `item.link(trace, run_name)` |
| Run-level | Final summary trace tagged with `run_name`, output = aggregate dict; Langfuse UI also auto-computes per-run means from dataset items |

**Dataset name:** derived from golden set stem (e.g., `golden_20260313_draft_v1`). Items are upserted on each run using `lf.create_dataset_item()`.  
**Run name:** `{experiment}_{YYYYMMDD_HHMMSS}` — timestamp ensures uniqueness.

## Makefile Target

```makefile
eval-chunking:
	python evals/eval_chunking.py \
		--golden-set $(GOLDEN) \
		--experiment $(EXP) \
		$(if $(RAGAS),--ragas,) \
		$(if $(TOP_K),--top-k $(TOP_K),) \
		$(if $(OUTPUT),--output $(OUTPUT),)
```

Usage:
```bash
make eval-chunking GOLDEN=tamu_data/evals/golden_sets/golden_20260313_draft_v1_sample10.jsonl EXP=chunk_600_ov100
make eval-chunking GOLDEN=... EXP=chunk_300_ov50 RAGAS=1
```

## Verification

1. Smoke test (no RAGAS): `python evals/eval_chunking.py --golden-set tamu_data/evals/golden_sets/golden_20260313_draft_v1_sample10.jsonl --experiment smoke`
   - Console shows per-query precision, hit_rate, retrieved_tokens + summary table
   - Langfuse shows 10 traces tagged `smoke_*`, each with 3 scores
2. Full run: same with full 50-question set
3. RAGAS run: add `--ragas` — console adds recall_at_k, f1_at_k; Langfuse traces gain 4 scores
4. JSON output: `--output` writes file with `items[]` + `aggregates{}` keys
5. Langfuse UI: dataset run view shows columns for each metric, comparable across experiment names
