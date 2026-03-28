# Design: Eval Skill + Benchmark Improvements

**Date:** 2026-03-18
**Status:** Approved

---

## Overview

Three coordinated changes:
1. Expose per-node `timing_ms` from `run_pipeline_v4()`
2. Enrich benchmark report with per-node latency breakdown + recurrent path visibility + correct `recurrent` flag to generator
3. Create `run-eval` skill for guided benchmark runs with automatic cache clearing

---

## 1. Pipeline Change — `run_pipeline_v4()`

**File:** `rag/v4/pipeline_v4.py`

Add optional `return_timing: bool = False` parameter. When `True`, return a 6-tuple:

```python
(chunks, router_result, data_gaps, data_integrity, conflicted_ids, timing_ms)
```

When `False` (default), returns the existing 5-tuple — fully backwards compatible.

`timing_ms` is already present in the LangGraph result dict (populated by `@timing_middleware` on every node). No new instrumentation needed — just expose it.

Type annotation for the function becomes a `Union` of 5-tuple and 6-tuple (or `tuple` with `overload` decorators). For simplicity, use a single return type of `tuple` with a comment noting the conditional 6th element.

**Callers to update:**
- `evals/run_benchmark.py` — pass `return_timing=True`, unpack 6-tuple
- `evals/run_probe.py` — no change needed (uses default `False`)

---

## 2. Benchmark Changes — `evals/run_benchmark.py`

### 2a. Fix recurrent flag in generator call

Currently hardcoded `recurrent=False`. Fix:

```python
recurrent=(rr.function == "recurrent")
```

This ensures the generator uses the correct prompt template when the recurrent path was taken.

### 2b. New `BenchmarkRow` fields

```python
is_recurrent: bool                   # True when router_function == "recurrent"
router_ms: Optional[float]           # timing_ms["router_node"] — always present
retrieval_ms: Optional[float]        # timing_ms["retrieval_node"] — always present
generator_node_ms: Optional[float]   # timing_ms["generator_node"] — always present (inside-graph time)
anchor_ms: Optional[float]           # timing_ms["anchor_node"] — recurrent only
eval_search_ms: Optional[float]      # timing_ms["eval_search_node"] — recurrent only
schedule_filter_ms: Optional[float]  # timing_ms["schedule_filter_node"] — recurrent only
merge_ms: Optional[float]            # timing_ms["merge_node"] — recurrent only
```

**Key naming:** `timing_ms` keys are the decorated function `__name__` values (e.g. `router_node`, `retrieval_node`). The `BenchmarkRow` fields use `*_ms` suffix for readability; the extraction maps `timing_ms["router_node"]` → `router_ms` etc.

**Two timing systems coexist:**
- `pipeline_ms`, `generator_ms`, `total_ms` — wall-clock from benchmark's outer timer (unchanged)
- `router_ms`, `retrieval_ms`, `generator_node_ms`, `anchor_ms`, etc. — per-node inside-graph time from `timing_ms`

Recurrent-only fields are `None` for non-recurrent rows. `generator_node_ms` is `None` for `out_of_scope` rows.

### 2c. Extraction logic in `run_one()`

After unpacking the 6-tuple:

```python
chunks, rr_result, data_gaps, data_integrity, conflicted_ids, timing_ms = run_pipeline_v4(query, return_timing=True)
```

Extract node timings:

```python
router_ms        = timing_ms.get("router_node")
retrieval_ms     = timing_ms.get("retrieval_node")
generator_node_ms = timing_ms.get("generator_node")
is_recurrent     = (rr.function == "recurrent")
anchor_ms        = timing_ms.get("anchor_node") if is_recurrent else None
eval_search_ms   = timing_ms.get("eval_search_node") if is_recurrent else None
schedule_filter_ms = timing_ms.get("schedule_filter_node") if is_recurrent else None
merge_ms         = timing_ms.get("merge_node") if is_recurrent else None
```

Existing fields `pipeline_ms`, `generator_ms`, `total_ms` (wall-clock from outer timer) are kept unchanged.

### 2c. Excel Per-Query tab — new columns

Inserted between `total_ms` and `answer_preview` in the `pq_cols` list:

| Column | Width | When populated |
|--------|-------|----------------|
| `is_recurrent` | 10 | always |
| `router_ms` | 12 | always |
| `retrieval_ms` | 12 | always |
| `generator_node_ms` | 16 | always (except out_of_scope) |
| `anchor_ms` | 12 | recurrent rows only (blank otherwise) |
| `eval_search_ms` | 14 | recurrent rows only |
| `schedule_filter_ms` | 16 | recurrent rows only |
| `merge_ms` | 10 | recurrent rows only |

### 2d. Excel Summary tab — new rows

Added after existing latency rows:
- `Mean router latency (ms)`
- `Mean retrieval latency (ms)`
- `Recurrent questions` — count + % of total (e.g. `3 (30%)`)

### 2e. Markdown report

Updated latency summary table gains `router_ms` and `retrieval_ms` mean rows.

When `any(r.is_recurrent for r in rows)`, append a **Recurrent Path Breakdown** section:

```markdown
## Recurrent Path Breakdown

N of M questions routed through recurrent path.

| Node | Mean (ms) |
|------|-----------|
| anchor | X |
| eval_search | X |
| retrieval | X |
| schedule_filter | X |
| merge | X |
```

### 2f. Column Definitions tab

Add entries for: `is_recurrent`, `router_ms`, `retrieval_ms`, `generator_node_ms`, `anchor_ms`, `eval_search_ms`, `schedule_filter_ms`, `merge_ms`.

Note: column names use `*_ms` suffix (user-facing); the underlying `timing_ms` dict keys are `*_node` (e.g. `timing_ms["router_node"]` → column `router_ms`).

---

## 3. Eval Skill — `.claude/skills/run-eval.md`

### Purpose

Guided benchmark runner: picks golden set, proposes experiment name, confirms all settings, clears sandbox cache, runs benchmark.

### Flow

**Step 1 — Discover golden sets**
List `tamu_data/evals/golden_sets/*.jsonl` files. Display numbered.

**Step 2 — Present all settings in one block**

```
Golden set:       [N] <filename>
Experiment name:  <stem>_v4_YYYYMMDD   ← auto-proposed, user can edit
RAGAS:            no
Cache clear:      yes (sandbox-down + sandbox-up, ~25s)

Confirm? [enter to run / edit above]
```

All settings are shown together to minimize back-and-forth. User confirms or requests changes.

**Step 3 — Clear cache (always)**

```bash
make sandbox-down && make sandbox-up
```

No prompt — happens every run.

**Step 4 — Run benchmark**

```bash
docker exec tamubot-claude-1 bash -c "cd /workspace && python evals/run_benchmark.py \
  --golden-set <path> --experiment-name <name> [--ragas]"
```

**Step 5 — Print results**

Show router accuracy, error count, and paths to `.xlsx` / `.md` reports. Remind about RAGAS validation if `--ragas` was not used.

### Experiment name convention

`{golden_stem}_v4_{YYYYMMDD}` — e.g. `golden_20260313_draft_v1_sample10_v4_20260318`

---

## Out of Scope

- No changes to `run_probe.py` (probe is ad-hoc, not a workflow skill target)
- No changes to `run_pipeline_v4_with_memory()` (conversation path, different use case)
- No parallelization of benchmark rows

---

## File Checklist

| File | Change |
|------|--------|
| `rag/v4/pipeline_v4.py` | Add `return_timing` param, expose `timing_ms` in return |
| `evals/run_benchmark.py` | New fields, recurrent flag fix, Excel/MD output changes |
| `.claude/skills/run-eval.md` | New skill file |
