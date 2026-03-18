# Run-Eval Skill + Benchmark Per-Node Breakdown — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose per-node pipeline timing in benchmark reports, fix the recurrent generator flag, and create a guided `run-eval` skill that resets the sandbox before every run.

**Architecture:** Three independent changes — (1) add `return_timing` param to `run_pipeline_v4()` to surface already-populated `timing_ms` dict; (2) update `run_benchmark.py` to capture and report per-node times + fix hardcoded `recurrent=False`; (3) write `.claude/skills/run-eval.md` skill file for guided UX.

**Tech Stack:** Python 3.14, LangGraph, openpyxl, Docker (sandbox-down/up), Claude skill markdown

**Spec:** `docs/superpowers/specs/2026-03-18-run-eval-design.md`

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `rag/v4/pipeline_v4.py` | Modify | Add `return_timing` param; expose `timing_ms` as 6th return |
| `evals/run_benchmark.py` | Modify | New `BenchmarkRow` fields; `run_one()` extraction; fix `recurrent` flag; Excel + MD output |
| `.claude/skills/run-eval.md` | Create | Guided benchmark skill |
| `tests/test_v4_pipeline_timing.py` | Create | Unit tests for `return_timing` behaviour |

---

## Task 1: Expose `timing_ms` from `run_pipeline_v4()`

**Files:**
- Modify: `rag/v4/pipeline_v4.py`
- Create: `tests/test_v4_pipeline_timing.py`

The `timing_ms` dict is already fully populated by `@timing_middleware` on every node and lives in the LangGraph result. This task just surfaces it when `return_timing=True`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_v4_pipeline_timing.py`:

```python
"""Tests for run_pipeline_v4 return_timing parameter."""
from unittest.mock import MagicMock, patch


def _make_mock_result(function: str = "hybrid_course") -> dict:
    """Minimal LangGraph result dict with timing_ms populated."""
    from rag.router import RouterResult
    rr = RouterResult(
        course_ids=["202611_CSCE_221_500"],
        rewritten_query="test",
        function=function,
    )
    return {
        "retrieved_chunks": [{"content": "chunk"}],
        "router_result": rr,
        "data_gaps": [],
        "data_integrity": True,
        "conflicted_course_ids": [],
        "timing_ms": {
            "router_node": 12.3,
            "retrieval_node": 45.6,
            "generator_node": 78.9,
        },
    }


def test_run_pipeline_v4_default_returns_5tuple():
    """Default call (no return_timing) still returns 5-tuple."""
    mock_result = _make_mock_result()
    with patch("rag.v4.pipeline_v4._get_graph") as mock_get_graph:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_result
        mock_get_graph.return_value = mock_graph

        from rag.v4.pipeline_v4 import run_pipeline_v4
        result = run_pipeline_v4("test query")

    assert len(result) == 5


def test_run_pipeline_v4_return_timing_true_returns_6tuple():
    """return_timing=True appends timing_ms dict as 6th element."""
    mock_result = _make_mock_result()
    with patch("rag.v4.pipeline_v4._get_graph") as mock_get_graph:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_result
        mock_get_graph.return_value = mock_graph

        from rag.v4.pipeline_v4 import run_pipeline_v4
        result = run_pipeline_v4("test query", return_timing=True)

    assert len(result) == 6
    timing = result[5]
    assert isinstance(timing, dict)
    assert "router_node" in timing
    assert timing["router_node"] == 12.3


def test_run_pipeline_v4_return_timing_empty_dict_on_missing():
    """If timing_ms is absent from result, returns empty dict (not KeyError)."""
    mock_result = _make_mock_result()
    mock_result.pop("timing_ms")
    with patch("rag.v4.pipeline_v4._get_graph") as mock_get_graph:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_result
        mock_get_graph.return_value = mock_graph

        from rag.v4.pipeline_v4 import run_pipeline_v4
        result = run_pipeline_v4("test query", return_timing=True)

    assert result[5] == {}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
docker exec tamubot-claude-1 bash -c "cd /workspace && python -m pytest tests/test_v4_pipeline_timing.py -v 2>&1"
```

Expected: 3 failures (function signature doesn't have `return_timing` yet).

- [ ] **Step 3: Implement `return_timing` in `run_pipeline_v4()`**

In `rag/v4/pipeline_v4.py`, update the `run_pipeline_v4` function signature and return:

```python
def run_pipeline_v4(
    query: str,
    trace=None,
    return_timing: bool = False,
) -> tuple:
    """Run the v4 pipeline.

    Args:
        query: User query string
        trace: Optional Langfuse trace
        return_timing: If True, returns 6-tuple with timing_ms dict appended.
                       Default False preserves backwards-compatible 5-tuple.

    Returns:
        (chunks, router_result, data_gaps, data_integrity, conflicted_course_ids)
        or if return_timing=True:
        (chunks, router_result, data_gaps, data_integrity, conflicted_course_ids, timing_ms)
    """
    initial_state: PipelineState = {
        "query": query,
        "trace": trace,
        "node_trace": [],
        "timing_ms": {},
        "conflicted_course_ids": [],
        "data_gaps": [],
        "data_integrity": True,
        "anchor_chunks": [],
        "discovery_chunks": [],
        "retrieved_chunks": [],
    }

    graph = _get_graph()
    result = graph.invoke(initial_state)

    five_tuple = (
        result.get("retrieved_chunks", []),
        result.get("router_result"),
        result.get("data_gaps", []),
        result.get("data_integrity", True),
        result.get("conflicted_course_ids", []),
    )

    if return_timing:
        return (*five_tuple, result.get("timing_ms", {}))
    return five_tuple
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
docker exec tamubot-claude-1 bash -c "cd /workspace && python -m pytest tests/test_v4_pipeline_timing.py -v 2>&1"
```

Expected: 3 passes.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
docker exec tamubot-claude-1 bash -c "cd /workspace && python -m pytest tests/ -v 2>&1"
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add rag/v4/pipeline_v4.py tests/test_v4_pipeline_timing.py
git commit -m "feat: expose timing_ms from run_pipeline_v4 via return_timing param"
```

---

## Task 2: Update `BenchmarkRow` + `run_one()`

**Files:**
- Modify: `evals/run_benchmark.py` (dataclass + `run_one()` only — output functions handled in Task 3)

This task adds the new fields to the dataclass, updates `run_one()` to unpack the 6-tuple, extracts per-node timings, and fixes the hardcoded `recurrent=False` bug.

- [ ] **Step 1: Add new fields to `BenchmarkRow` dataclass**

In `evals/run_benchmark.py`, after the existing timing fields (`pipeline_ms`, `generator_ms`, `total_ms`), add:

```python
    # Per-node timing from pipeline timing_ms (inside-graph, not wall-clock)
    is_recurrent: bool = False
    router_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    generator_node_ms: Optional[float] = None
    # Recurrent path only (None for non-recurrent rows)
    anchor_ms: Optional[float] = None
    eval_search_ms: Optional[float] = None
    schedule_filter_ms: Optional[float] = None
    merge_ms: Optional[float] = None
```

Use `= None` / `= False` defaults so the dataclass doesn't break positional instantiation in existing tests.

- [ ] **Step 2: Update `run_one()` — unpack 6-tuple and extract timings**

Replace the `run_pipeline_v4(query)` call and its result handling block:

```python
    # Router + Retrieval (v4 graph)
    timing_ms: dict = {}
    try:
        chunks, rr_result, data_gaps, data_integrity, conflicted_ids, timing_ms = run_pipeline_v4(
            query, return_timing=True
        )
        if rr_result is not None:
            rr = rr_result
        else:
            error = "retrieval: router_result is None (pipeline did not complete)"
    except Exception as e:
        error = f"retrieval: {e}"
    pipeline_ms = round((time.perf_counter() - t0) * 1000, 1)
```

- [ ] **Step 3: Extract per-node timings in `run_one()`**

After the pipeline try/except block, before the generation block, add:

```python
    # Per-node timing extraction
    is_recurrent = (rr.function == "recurrent")
    router_ms = timing_ms.get("router_node")
    retrieval_ms = timing_ms.get("retrieval_node")
    generator_node_ms = timing_ms.get("generator_node")
    anchor_ms = timing_ms.get("anchor_node") if is_recurrent else None
    eval_search_ms = timing_ms.get("eval_search_node") if is_recurrent else None
    schedule_filter_ms = timing_ms.get("schedule_filter_node") if is_recurrent else None
    merge_ms = timing_ms.get("merge_node") if is_recurrent else None
```

- [ ] **Step 4: Fix hardcoded `recurrent=False` in `generator_order()` call**

Change:

```python
            stream = generator_order(
                recurrent=False,
```

To:

```python
            stream = generator_order(
                recurrent=is_recurrent,
```

- [ ] **Step 5: Add new fields to the `BenchmarkRow(...)` constructor call**

In `run_one()`, add the new fields to the return statement:

```python
        is_recurrent=is_recurrent,
        router_ms=router_ms,
        retrieval_ms=retrieval_ms,
        generator_node_ms=generator_node_ms,
        anchor_ms=anchor_ms,
        eval_search_ms=eval_search_ms,
        schedule_filter_ms=schedule_filter_ms,
        merge_ms=merge_ms,
```

- [ ] **Step 6: Smoke-test with a dry run (no API calls)**

```bash
docker exec tamubot-claude-1 bash -c "cd /workspace && python -c \"
from evals.run_benchmark import BenchmarkRow
r = BenchmarkRow(
    question_id=1, question='test', stratum='s', source_course_id='CSCE 670',
    expected_function='hybrid_course', router_function='hybrid_course',
    router_function_correct=True, router_rewritten_query='test',
    router_course_ids='CSCE 670', router_intent_type='ACADEMIC',
    chunks_retrieved=5, est_input_tokens=100, est_output_tokens=50,
    pipeline_ms=100.0, generator_ms=50.0, total_ms=150.0,
    answer_full='answer', answer_preview='answer', citation_pass=True,
    is_recurrent=False, router_ms=12.3, retrieval_ms=45.6,
    generator_node_ms=78.9,
)
print('BenchmarkRow OK:', r.router_ms, r.is_recurrent)
\" 2>&1"
```

Expected: `BenchmarkRow OK: 12.3 False`

- [ ] **Step 7: Commit**

```bash
git add evals/run_benchmark.py
git commit -m "feat: add per-node timing fields to BenchmarkRow and fix recurrent generator flag"
```

---

## Task 3: Update Excel + Markdown Output

**Files:**
- Modify: `evals/run_benchmark.py` (`write_excel()` and `write_markdown()` functions)

- [ ] **Step 1: Update `pq_cols` in `write_excel()` — add new timing columns**

After the `("total_ms", 10)` entry in the `pq_cols` list and before `("answer_preview", 45)`, insert:

```python
        # Per-node timing (inside-graph)
        ("is_recurrent", 10),
        ("router_ms", 12),
        ("retrieval_ms", 12),
        ("generator_node_ms", 16),
        ("anchor_ms", 12),
        ("eval_search_ms", 14),
        ("schedule_filter_ms", 16),
        ("merge_ms", 10),
```

- [ ] **Step 2: Update the per-query `values` list in `write_excel()` to match**

In the row-writing loop, add the new values after `r.total_ms` and before `r.answer_preview`:

```python
            r.is_recurrent, r.router_ms, r.retrieval_ms, r.generator_node_ms,
            r.anchor_ms, r.eval_search_ms, r.schedule_filter_ms, r.merge_ms,
```

- [ ] **Step 3: Update Summary tab in `write_excel()`**

After `("Mean generator latency (ms)", _avg(rows, "generator_ms"))` in `summary_rows`, add:

```python
        ("Mean router latency (ms)", _avg(rows, "router_ms")),
        ("Mean retrieval latency (ms)", _avg(rows, "retrieval_ms")),
        ("Recurrent questions",
         f"{sum(1 for r in rows if r.is_recurrent)} ({sum(1 for r in rows if r.is_recurrent)/n:.0%})"
         if n else "N/A"),
```

- [ ] **Step 4: Update Column Definitions tab in `write_excel()`**

After the existing `generator_ms` entry in `col_defs`, add:

```python
        ("is_recurrent", "bool", "True when the router selected the recurrent function. Recurrent queries span the full corpus looking for topic-adjacent courses rather than fetching from a specific course."),
        ("router_ms", "float (ms)", "Inside-graph time for the router node (LLM call that classifies function and rewrites query). Sourced from timing_ms['router_node'] — more precise than wall-clock pipeline_ms."),
        ("retrieval_ms", "float (ms)", "Inside-graph time for the retrieval node (embedding + MongoDB search + reranking). For recurrent queries this covers the cross-corpus search only."),
        ("generator_node_ms", "float (ms)", "Inside-graph time for the generator node (context assembly + LLM streaming). None for out_of_scope rows. Compare to wall-clock generator_ms — difference is Python overhead."),
        ("anchor_ms", "float (ms) / blank", "Recurrent path only. Time for anchor_node: fetches schedule and meeting-time anchor chunks for each detected course."),
        ("eval_search_ms", "float (ms) / blank", "Recurrent path only. Time for eval_search_node: generates an expanded search string via LLM for cross-corpus discovery."),
        ("schedule_filter_ms", "float (ms) / blank", "Recurrent path only. Time for schedule_filter_node: filters discovery chunks to remove schedule conflicts."),
        ("merge_ms", "float (ms) / blank", "Recurrent path only. Time for merge_node: deduplicates and merges anchor + discovery chunks into retrieved_chunks."),
```

- [ ] **Step 5: Update `write_markdown()` — latency table**

After the `Mean generator latency (ms)` row in the markdown summary table, add:

```python
            f"| Mean router latency (ms) | {_fmt_ms('router_ms')} |",
            f"| Mean retrieval latency (ms) | {_fmt_ms('retrieval_ms')} |",
```

- [ ] **Step 6: Add Recurrent Path Breakdown section to `write_markdown()`**

After the errors section, add:

```python
    # Recurrent path breakdown (only when recurrent rows exist)
    recurrent_rows = [r for r in rows if r.is_recurrent]
    if recurrent_rows:
        n_rec = len(recurrent_rows)

        def _rec_avg(attr: str) -> str:
            vals = [getattr(r, attr) for r in recurrent_rows if getattr(r, attr) is not None]
            return f"{sum(vals)/len(vals):.0f}" if vals else "N/A"

        lines += [
            "",
            f"## Recurrent Path Breakdown",
            "",
            f"{n_rec} of {n} questions routed through recurrent path.",
            "",
            "| Node | Mean (ms) |",
            "|------|-----------|",
            f"| anchor | {_rec_avg('anchor_ms')} |",
            f"| eval_search | {_rec_avg('eval_search_ms')} |",
            f"| retrieval | {_rec_avg('retrieval_ms')} |",
            f"| schedule_filter | {_rec_avg('schedule_filter_ms')} |",
            f"| merge | {_rec_avg('merge_ms')} |",
        ]
```

- [ ] **Step 7: Run lint and typecheck**

```bash
docker exec tamubot-claude-1 bash -c "cd /workspace && python -m ruff check evals/run_benchmark.py 2>&1"
```

Expected: no errors.

- [ ] **Step 8: Run a full benchmark to verify output**

```bash
docker exec tamubot-claude-1 bash -c "cd /workspace && python evals/run_benchmark.py \
  --golden-set tamu_data/evals/golden_sets/golden_20260313_draft_v1_sample10.jsonl \
  --experiment-name breakdown_smoke_$(date +%Y%m%d) 2>&1"
```

Expected: benchmark completes, reports written, new timing columns visible.

- [ ] **Step 9: Commit**

```bash
git add evals/run_benchmark.py
git commit -m "feat: add per-node timing to benchmark Excel/MD output and recurrent breakdown section"
```

---

## Task 4: Create `run-eval` Skill

**Files:**
- Create: `.claude/skills/run-eval.md`

This is a markdown skill file — no tests needed. The skill is invoked by Claude when the user says "run eval", "benchmark", or similar.

- [ ] **Step 1: Create `.claude/skills/run-eval.md`**

```markdown
---
description: Guided benchmark runner — lists golden sets, proposes experiment name, clears sandbox cache, runs benchmark
triggers: ["run eval", "run benchmark", "run evals", "benchmark the pipeline", "benchmark rag"]
---

# Run Eval Skill

Announce: "Using run-eval skill to set up and run the benchmark."

## Step 1 — Discover golden sets

List all `.jsonl` files in `tamu_data/evals/golden_sets/`:

```bash
ls tamu_data/evals/golden_sets/*.jsonl
```

Display them numbered, e.g.:

```
Available golden sets:
  [1] golden_20260313_draft_v1_sample10.jsonl   (10 questions)
  [2] golden_20260313_draft_v1.jsonl            (60 questions)
```

Show question count for each: `wc -l <file>`.

## Step 2 — Present all settings in one block

Propose settings and show them all together. Use today's date (YYYYMMDD). Default golden set = most recently modified file. Default experiment name = `{stem}_v4_{YYYYMMDD}`.

```
Ready to run benchmark:

  Golden set:       [1] golden_20260313_draft_v1_sample10.jsonl
  Experiment name:  golden_20260313_draft_v1_sample10_v4_20260318
  RAGAS:            no
  Cache clear:      yes (sandbox-down + sandbox-up, ~25s)

Confirm, or tell me what to change.
```

Wait for user confirmation. If user edits any field, update and confirm again before proceeding.

## Step 3 — Clear sandbox cache (always, no prompt)

```bash
make sandbox-down && make sandbox-up
```

Wait for both commands to complete before proceeding.

## Step 4 — Run benchmark

```bash
docker exec tamubot-claude-1 bash -c "cd /workspace && python evals/run_benchmark.py \
  --golden-set tamu_data/evals/golden_sets/<golden_file> \
  --experiment-name <experiment_name> [--ragas]"
```

Stream output to user as it runs.

## Step 5 — Show results

After completion, print:
- Router accuracy
- Error count (if any)
- Paths to `.xlsx` and `.md` reports

If RAGAS was **not** run, remind:
```
To add RAGAS scores later:
  python evals/validate_ragas.py --benchmark <xlsx_path>
```
```

- [ ] **Step 2: Verify skill file is valid**

```bash
head -5 .claude/skills/run-eval.md
```

Expected: frontmatter block visible.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/run-eval.md
git commit -m "feat: add run-eval skill for guided benchmark runs with cache clearing"
```

---

## Final Verification

- [ ] **Run full test suite**

```bash
docker exec tamubot-claude-1 bash -c "cd /workspace && python -m pytest tests/ -v 2>&1"
```

Expected: all tests pass including `test_v4_pipeline_timing.py`.

- [ ] **Run lint + typecheck**

```bash
docker exec tamubot-claude-1 bash -c "cd /workspace && make lint && make typecheck 2>&1"
```

Expected: no errors.

- [ ] **End-to-end benchmark smoke test**

```bash
docker exec tamubot-claude-1 bash -c "cd /workspace && python evals/run_benchmark.py \
  --golden-set tamu_data/evals/golden_sets/golden_20260313_draft_v1_sample10.jsonl \
  --experiment-name final_smoke_$(date +%Y%m%d) 2>&1"
```

Expected: 10/10 questions, new timing columns in output, no errors.
