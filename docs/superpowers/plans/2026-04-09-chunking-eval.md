# Chunking Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a retrieval-only benchmark script (`evals/eval_chunking.py`) that measures precision, recall, F1, hit-rate, and token efficiency per query, and logs results to Langfuse for experiment comparison.

**Architecture:** A new standalone script runs retrieval (router → hybrid/semantic search → rerank) without the generator. Embedding-based metrics (precision, hit-rate, tokens) are always computed via Voyage-3 cosine similarity. RAGAS ContextPrecision + ContextRecall are computed only when `--ragas` is passed. Per-query traces and a run-level aggregate trace are logged to Langfuse.

**Tech Stack:** Python, `rag` public API (`classify_query`, `hybrid_search`, `search_semantic`, `rerank`, `compute_dynamic_k`), `evals/eval_retrieval_metrics.label_relevant`, `ragas` (ContextPrecision, ContextRecall), Langfuse Python SDK (existing `get_langfuse()` singleton).

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `rag/tools/langfuse.py` | Modify | Add `compute_retrieval_ragas()` — RAGAS retrieval metrics |
| `evals/eval_chunking.py` | Create | Main eval script: loader, metrics, retrieval, loop, CLI |
| `tests/test_eval_chunking.py` | Create | Unit tests for pure functions in eval_chunking.py |
| `tests/test_langfuse_retrieval_ragas.py` | Create | Unit tests for compute_retrieval_ragas() |
| `Makefile` | Modify | Add `eval-chunking` target |

---

## Task 1: Add `compute_retrieval_ragas()` to `rag/tools/langfuse.py`

**Files:**
- Modify: `rag/tools/langfuse.py` (append after `run_groundedness_scoring_background`)
- Create: `tests/test_langfuse_retrieval_ragas.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_langfuse_retrieval_ragas.py`:

```python
"""Unit tests for compute_retrieval_ragas in rag/tools/langfuse.py."""
import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_compute_retrieval_ragas_returns_empty_on_exception():
    """Returns {} when any import or RAGAS call fails."""
    with patch("rag.tools.langfuse.get_langfuse", return_value=None), \
         patch.dict("sys.modules", {"langchain_openai": None}):
        from rag.tools.langfuse import compute_retrieval_ragas
        result = compute_retrieval_ragas(
            question="What is grading?",
            contexts=["Grading info"],
            reference="The course uses letter grades.",
        )
    assert result == {}


def test_compute_retrieval_ragas_returns_scores():
    """Returns context_precision and context_recall on success."""
    mock_df = pd.DataFrame([{"context_precision": 0.75, "context_recall": 0.8}])
    mock_eval_result = MagicMock()
    mock_eval_result.to_pandas.return_value = mock_df

    mock_llm = MagicMock()
    mock_wrapper = MagicMock()
    mock_sample = MagicMock()
    mock_dataset = MagicMock()
    mock_metric = MagicMock()

    with patch("rag.tools.langfuse.get_langfuse", return_value=None), \
         patch("langchain_openai.ChatOpenAI", return_value=mock_llm), \
         patch("ragas.llms.LangchainLLMWrapper", return_value=mock_wrapper), \
         patch("ragas.SingleTurnSample", return_value=mock_sample), \
         patch("ragas.EvaluationDataset", return_value=mock_dataset), \
         patch("ragas.metrics.ContextPrecision", return_value=mock_metric), \
         patch("ragas.metrics.ContextRecall", return_value=mock_metric), \
         patch("ragas.evaluate", return_value=mock_eval_result):
        from rag.tools.langfuse import compute_retrieval_ragas
        result = compute_retrieval_ragas(
            question="What is grading?",
            contexts=["Grading info"],
            reference="Letter grades.",
        )

    assert result.get("context_precision") == 0.75
    assert result.get("context_recall") == 0.8


def test_compute_retrieval_ragas_uploads_scores_to_langfuse():
    """Calls lf.create_score() for each metric when trace_id is provided."""
    mock_df = pd.DataFrame([{"context_precision": 0.6, "context_recall": 0.7}])
    mock_eval_result = MagicMock()
    mock_eval_result.to_pandas.return_value = mock_df

    mock_lf = MagicMock()
    mock_llm = MagicMock()
    mock_metric = MagicMock()

    with patch("rag.tools.langfuse.get_langfuse", return_value=mock_lf), \
         patch("langchain_openai.ChatOpenAI", return_value=mock_llm), \
         patch("ragas.llms.LangchainLLMWrapper"), \
         patch("ragas.SingleTurnSample"), \
         patch("ragas.EvaluationDataset"), \
         patch("ragas.metrics.ContextPrecision", return_value=mock_metric), \
         patch("ragas.metrics.ContextRecall", return_value=mock_metric), \
         patch("ragas.evaluate", return_value=mock_eval_result):
        from rag.tools.langfuse import compute_retrieval_ragas
        compute_retrieval_ragas(
            question="query",
            contexts=["ctx"],
            reference="ref",
            trace_id="trace-abc",
        )

    assert mock_lf.create_score.call_count == 2
    calls = {c.kwargs["name"] for c in mock_lf.create_score.call_args_list}
    assert "context_precision" in calls
    assert "context_recall" in calls
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd /workspace && python -m pytest tests/test_langfuse_retrieval_ragas.py -v 2>&1 | tail -20
```

Expected: `ImportError` or `AttributeError` (function not defined yet).

- [ ] **Step 3: Implement `compute_retrieval_ragas()`**

Append to `rag/tools/langfuse.py` (after `run_groundedness_scoring_background`, before the end of file):

```python


def compute_retrieval_ragas(
    question: str,
    contexts: list[str],
    reference: str,
    trace_id: Optional[str] = None,
) -> dict:
    """Compute RAGAS ContextPrecision + ContextRecall for retrieval evaluation.

    Unlike compute_ragas_metrics() (which needs a generated answer), this
    function uses only the retrieved chunks and the golden reference answer —
    no generator call required.

    Args:
        question:  The user query.
        contexts:  List of retrieved chunk text strings.
        reference: The reference answer from the golden set.
        trace_id:  Langfuse trace ID to attach scores to (optional).

    Returns:
        Dict with keys 'context_precision' and 'context_recall' (floats in
        [0, 1]), or {} on failure.
    """
    try:
        from langchain_openai import ChatOpenAI
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import ContextPrecision, ContextRecall

        import config

        critic_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=config.TAMU_MODEL,
                api_key=config.TAMU_API_KEY,
                base_url=config.TAMU_BASE_URL,
                temperature=0,
            )
        )

        try:
            import litellm
            litellm.max_budget = None
        except Exception:
            pass

        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            reference=reference,
        )
        dataset = EvaluationDataset(samples=[sample])
        metrics = [ContextPrecision(llm=critic_llm), ContextRecall(llm=critic_llm)]

        result = evaluate(dataset=dataset, metrics=metrics)
        scores: dict = result.to_pandas().iloc[0].to_dict()

        import math
        clean = {
            k: round(float(v), 4)
            for k, v in scores.items()
            if isinstance(v, (int, float)) and not math.isnan(v)
        }

        lf = get_langfuse()
        if lf and trace_id:
            for name, value in clean.items():
                lf.create_score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    comment="RAGAS retrieval evaluation",
                )

        logger.info(f"Retrieval RAGAS scores for trace {trace_id}: {clean}")
        return clean

    except Exception as e:
        logger.warning(f"Retrieval RAGAS evaluation failed: {e}")
        return {}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /workspace && python -m pytest tests/test_langfuse_retrieval_ragas.py -v 2>&1 | tail -20
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
cd /workspace && git add rag/tools/langfuse.py tests/test_langfuse_retrieval_ragas.py
git commit -m "feat(evals): add compute_retrieval_ragas for retrieval-only RAGAS metrics"
```

---

## Task 2: Create `evals/eval_chunking.py` — golden set loader + embedding metrics

**Files:**
- Create: `evals/eval_chunking.py`
- Create: `tests/test_eval_chunking.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_eval_chunking.py`:

```python
"""Unit tests for pure functions in evals/eval_chunking.py."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# load_golden_set
# ---------------------------------------------------------------------------

def test_load_golden_set_returns_all_items(tmp_path):
    """Reads every non-empty JSONL line."""
    data = [
        {"question": "What is grading?", "reference_answer": "A/B/C"},
        {"question": "Who teaches it?", "reference_answer": "Dr. Smith"},
    ]
    gs_file = tmp_path / "golden.jsonl"
    gs_file.write_text("\n".join(json.dumps(d) for d in data), encoding="utf-8")

    from evals.eval_chunking import load_golden_set
    result = load_golden_set(gs_file)

    assert len(result) == 2
    assert result[0]["question"] == "What is grading?"


def test_load_golden_set_skips_blank_lines(tmp_path):
    """Ignores blank lines in JSONL."""
    gs_file = tmp_path / "golden.jsonl"
    gs_file.write_text(
        '{"question": "Q1", "reference_answer": "A1"}\n\n\n',
        encoding="utf-8",
    )

    from evals.eval_chunking import load_golden_set
    result = load_golden_set(gs_file)

    assert len(result) == 1


# ---------------------------------------------------------------------------
# compute_embedding_metrics
# ---------------------------------------------------------------------------

def test_compute_embedding_metrics_all_relevant():
    """precision=1.0, hit_rate=1.0 when all chunks are relevant."""
    chunks = [{"content": "abc" * 10}, {"content": "def" * 20}]
    labels = [True, True]

    from evals.eval_chunking import compute_embedding_metrics
    result = compute_embedding_metrics("query", chunks, _labels=labels)

    assert result["precision_at_k"] == 1.0
    assert result["hit_rate_at_k"] == 1.0
    assert result["retrieved_tokens"] == (30 // 4) + (60 // 4)


def test_compute_embedding_metrics_none_relevant():
    """precision=0.0, hit_rate=0.0 when no chunks are relevant."""
    chunks = [{"content": "abc"}, {"content": "def"}]
    labels = [False, False]

    from evals.eval_chunking import compute_embedding_metrics
    result = compute_embedding_metrics("query", chunks, _labels=labels)

    assert result["precision_at_k"] == 0.0
    assert result["hit_rate_at_k"] == 0.0


def test_compute_embedding_metrics_partial_relevance():
    """precision=0.5 when 1 of 2 chunks is relevant."""
    chunks = [{"content": "a" * 100}, {"content": "b" * 100}]
    labels = [True, False]

    from evals.eval_chunking import compute_embedding_metrics
    result = compute_embedding_metrics("query", chunks, _labels=labels)

    assert result["precision_at_k"] == 0.5
    assert result["hit_rate_at_k"] == 1.0
    assert result["retrieved_tokens"] == (100 // 4) * 2


def test_compute_embedding_metrics_empty_chunks():
    """Returns zeros when chunk list is empty."""
    from evals.eval_chunking import compute_embedding_metrics
    result = compute_embedding_metrics("query", [], _labels=[])

    assert result["precision_at_k"] == 0.0
    assert result["hit_rate_at_k"] == 0.0
    assert result["retrieved_tokens"] == 0


# ---------------------------------------------------------------------------
# _compute_aggregates
# ---------------------------------------------------------------------------

def test_compute_aggregates_means():
    """Averages each numeric metric across results."""
    results = [
        {"precision_at_k": 0.5, "hit_rate_at_k": 1.0, "retrieved_tokens": 100,
         "recall_at_k": None, "f1_at_k": None, "context_precision": None},
        {"precision_at_k": 1.0, "hit_rate_at_k": 1.0, "retrieved_tokens": 200,
         "recall_at_k": None, "f1_at_k": None, "context_precision": None},
    ]

    from evals.eval_chunking import _compute_aggregates
    agg = _compute_aggregates(results)

    assert agg["avg_precision_at_k"] == 0.75
    assert agg["avg_hit_rate_at_k"] == 1.0
    assert agg["avg_retrieved_tokens"] == 150.0
    assert agg["n_queries"] == 2
    assert "avg_recall_at_k" not in agg  # all None → excluded


def test_compute_f1():
    """F1 is harmonic mean of precision and recall."""
    from evals.eval_chunking import _compute_f1
    assert _compute_f1(1.0, 1.0) == 1.0
    assert _compute_f1(0.0, 1.0) == 0.0
    assert _compute_f1(0.5, 0.5) == 0.5
    assert _compute_f1(0.0, 0.0) == 0.0
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd /workspace && python -m pytest tests/test_eval_chunking.py -v 2>&1 | tail -20
```

Expected: `ModuleNotFoundError: No module named 'evals.eval_chunking'`.

- [ ] **Step 3: Create `evals/eval_chunking.py` with loader + embedding metrics**

```python
"""Retrieval-only chunking benchmark.

Measures precision_at_k, recall_at_k (RAGAS ContextRecall), f1_at_k,
hit_rate_at_k, and retrieved_tokens per query. Logs per-query traces and
a run-level aggregate trace to Langfuse.

Usage:
    python evals/eval_chunking.py \\
        --golden-set tamu_data/evals/golden_sets/golden_20260313_draft_v1.jsonl \\
        [--experiment chunk_600_ov100] \\
        [--top-k 7] \\
        [--threshold 0.85] \\
        [--ragas] \\
        [--output tamu_data/evals/reports/chunking_YYYYMMDD.json]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402 — must come after sys.path insert

logger = logging.getLogger("tamubot.eval_chunking")


# ---------------------------------------------------------------------------
# Golden set loader
# ---------------------------------------------------------------------------

def load_golden_set(path: Path) -> list[dict]:
    """Load a golden set JSONL file. Returns list of question dicts."""
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Embedding-based metrics (cheap, always computed)
# ---------------------------------------------------------------------------

def compute_embedding_metrics(
    query: str,
    chunks: list[dict],
    threshold: float = 0.85,
    _labels: Optional[list[bool]] = None,
) -> dict:
    """Compute embedding-based retrieval metrics without an LLM.

    Args:
        query:     User query string.
        chunks:    Reranked chunk dicts (must have 'content' key).
        threshold: Voyage-3 cosine similarity threshold (default 0.85).
        _labels:   Pre-computed relevance labels (skips Voyage AI; for tests).

    Returns:
        Dict with keys: precision_at_k (float), hit_rate_at_k (float),
        retrieved_tokens (int).
    """
    if not chunks:
        return {"precision_at_k": 0.0, "hit_rate_at_k": 0.0, "retrieved_tokens": 0}

    from evals.eval_retrieval_metrics import label_relevant

    labels = _labels if _labels is not None else label_relevant(query, chunks, threshold)
    k = len(labels)
    n_relevant = sum(labels)

    return {
        "precision_at_k": round(n_relevant / k, 4) if k > 0 else 0.0,
        "hit_rate_at_k": 1.0 if n_relevant > 0 else 0.0,
        "retrieved_tokens": sum(len(c.get("content", "")) // 4 for c in chunks),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return round(2.0 * precision * recall / (precision + recall), 4)


def _compute_aggregates(results: list[dict]) -> dict:
    """Mean of each numeric metric across all query results."""
    metrics = [
        "precision_at_k", "hit_rate_at_k", "retrieved_tokens",
        "recall_at_k", "f1_at_k", "context_precision",
    ]
    aggregates: dict = {}
    for m in metrics:
        values = [r[m] for r in results if r.get(m) is not None]
        if values:
            aggregates[f"avg_{m}"] = round(sum(values) / len(values), 4)
    aggregates["n_queries"] = len(results)
    return aggregates
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /workspace && python -m pytest tests/test_eval_chunking.py -v 2>&1 | tail -20
```

Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
cd /workspace && git add evals/eval_chunking.py tests/test_eval_chunking.py
git commit -m "feat(evals): add eval_chunking.py scaffold with golden set loader and embedding metrics"
```

---

## Task 3: Add `retrieve_for_query()` to `evals/eval_chunking.py`

**Files:**
- Modify: `evals/eval_chunking.py` (append after `_compute_aggregates`)
- Modify: `tests/test_eval_chunking.py` (append tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_eval_chunking.py`:

```python
# ---------------------------------------------------------------------------
# retrieve_for_query
# ---------------------------------------------------------------------------

def test_retrieve_for_query_returns_empty_for_out_of_scope():
    """Returns [] when router function has no retrieval path."""
    mock_rr = MagicMock()
    mock_rr.function = "out_of_scope"
    mock_rr.course_ids = []
    mock_rr.rewritten_query = "query"

    with patch("evals.eval_chunking.compute_dynamic_k", return_value={"retrieve_k": 20, "rerank_k": 5}), \
         patch("evals.eval_chunking.hybrid_search") as mock_hs, \
         patch("evals.eval_chunking.search_semantic") as mock_sem, \
         patch("evals.eval_chunking.rerank") as mock_rr_fn:
        from evals.eval_chunking import retrieve_for_query
        result = retrieve_for_query("query", mock_rr, top_k=None)

    assert result == []
    mock_hs.assert_not_called()
    mock_sem.assert_not_called()


def test_retrieve_for_query_uses_hybrid_for_course_queries():
    """Calls hybrid_search when course_ids are present."""
    mock_rr = MagicMock()
    mock_rr.function = "hybrid_course"
    mock_rr.course_ids = ["202611_CSCE_638_500"]
    mock_rr.rewritten_query = "grading policy"

    fake_chunks = [{"content": "chunk1"}, {"content": "chunk2"}]
    fake_reranked = [{"content": "chunk1"}]

    with patch("evals.eval_chunking.compute_dynamic_k", return_value={"retrieve_k": 20, "rerank_k": 5}), \
         patch("evals.eval_chunking.hybrid_search", return_value=fake_chunks) as mock_hs, \
         patch("evals.eval_chunking.rerank", return_value=fake_reranked) as mock_rr_fn:
        from evals.eval_chunking import retrieve_for_query
        result = retrieve_for_query("grading query", mock_rr, top_k=None)

    mock_hs.assert_called_once_with(
        "grading policy",
        filters={"course_id": "202611_CSCE_638_500"},
        k=20,
    )
    mock_rr_fn.assert_called_once_with("grading policy", fake_chunks, top_k=5)
    assert result == fake_reranked


def test_retrieve_for_query_top_k_overrides_dynamic():
    """--top-k CLI flag overrides compute_dynamic_k rerank_k."""
    mock_rr = MagicMock()
    mock_rr.function = "hybrid_course"
    mock_rr.course_ids = ["202611_CSCE_638_500"]
    mock_rr.rewritten_query = "query"

    with patch("evals.eval_chunking.compute_dynamic_k", return_value={"retrieve_k": 20, "rerank_k": 5}), \
         patch("evals.eval_chunking.hybrid_search", return_value=[{"content": "c"}]), \
         patch("evals.eval_chunking.rerank", return_value=[]) as mock_rr_fn:
        from evals.eval_chunking import retrieve_for_query
        retrieve_for_query("query", mock_rr, top_k=3)

    mock_rr_fn.assert_called_once_with("query", [{"content": "c"}], top_k=3)


def test_retrieve_for_query_semantic_general():
    """Calls search_semantic (not hybrid_search) for semantic_general function."""
    mock_rr = MagicMock()
    mock_rr.function = "semantic_general"
    mock_rr.course_ids = []
    mock_rr.rewritten_query = "related courses"

    with patch("evals.eval_chunking.compute_dynamic_k", return_value={"retrieve_k": 30, "rerank_k": 10}), \
         patch("evals.eval_chunking.search_semantic", return_value=[{"content": "s"}]) as mock_sem, \
         patch("evals.eval_chunking.rerank", return_value=[{"content": "s"}]):
        from evals.eval_chunking import retrieve_for_query
        retrieve_for_query("semantic query", mock_rr, top_k=None)

    mock_sem.assert_called_once_with("related courses", top_k=30)
```

`MagicMock` and `patch` are already imported at the top of `tests/test_eval_chunking.py` from Task 2. No additional imports needed.

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd /workspace && python -m pytest tests/test_eval_chunking.py::test_retrieve_for_query_returns_empty_for_out_of_scope -v 2>&1 | tail -10
```

Expected: `ImportError` or `AttributeError` (`retrieve_for_query` not defined).

- [ ] **Step 3: Implement `retrieve_for_query()`**

Append to `evals/eval_chunking.py` (after `_compute_aggregates`):

```python
# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

from rag import classify_query, compute_dynamic_k, hybrid_search, rerank, search_semantic  # noqa: E402


def retrieve_for_query(
    query: str,
    rr,
    top_k: Optional[int] = None,
) -> list[dict]:
    """Run hybrid/semantic search + rerank for a single query.

    Args:
        query:  Original user query (used as fallback if rr.rewritten_query is empty).
        rr:     RouterResult from classify_query().
        top_k:  Override for rerank_k. If None, uses compute_dynamic_k().

    Returns:
        Reranked list of chunk dicts, or [] when the function has no retrieval
        path (e.g. out_of_scope, recurrent without course_ids).
    """
    dk = compute_dynamic_k(rr.function, len(rr.course_ids))
    retrieve_k = dk["retrieve_k"]
    rerank_k = top_k if top_k is not None else dk["rerank_k"]

    search_query = rr.rewritten_query or query

    if rr.function == "semantic_general":
        pre_results = search_semantic(search_query, top_k=retrieve_k)
    elif rr.course_ids:
        pre_results = hybrid_search(
            search_query,
            filters={"course_id": rr.course_ids[0]},
            k=retrieve_k,
        )
    else:
        return []

    return rerank(search_query, pre_results, top_k=rerank_k)
```

- [ ] **Step 4: Run all tests — expect PASS**

```bash
cd /workspace && python -m pytest tests/test_eval_chunking.py -v 2>&1 | tail -20
```

Expected: all tests pass (new + existing).

- [ ] **Step 5: Commit**

```bash
cd /workspace && git add evals/eval_chunking.py tests/test_eval_chunking.py
git commit -m "feat(evals): add retrieve_for_query to eval_chunking"
```

---

## Task 4: Add main eval loop + Langfuse logging to `evals/eval_chunking.py`

**Files:**
- Modify: `evals/eval_chunking.py` (append after `retrieve_for_query`)
- Modify: `tests/test_eval_chunking.py` (append tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_eval_chunking.py`:

```python
# ---------------------------------------------------------------------------
# run_eval (main loop)
# ---------------------------------------------------------------------------

def _make_item(question="What is grading?", reference="Letter grades."):
    return {"question": question, "reference_answer": reference}


def test_run_eval_skips_item_without_reference_when_ragas():
    """Items without reference_answer are skipped when --ragas is on."""
    items = [{"question": "Q1", "reference_answer": ""}]

    with patch("evals.eval_chunking.classify_query") as mock_cq:
        from evals.eval_chunking import run_eval
        results, _ = run_eval(items, "test", None, 0.85, ragas_enabled=True, lf=None)

    mock_cq.assert_not_called()
    assert results == []


def test_run_eval_skips_item_without_question():
    """Items missing 'question' key are skipped."""
    items = [{"reference_answer": "ref"}]

    with patch("evals.eval_chunking.classify_query") as mock_cq:
        from evals.eval_chunking import run_eval
        results, _ = run_eval(items, "test", None, 0.85, ragas_enabled=False, lf=None)

    mock_cq.assert_not_called()
    assert results == []


def test_run_eval_returns_embedding_metrics_only():
    """Without --ragas, returns precision, hit_rate, tokens per query."""
    mock_rr = MagicMock()
    mock_rr.function = "hybrid_course"
    mock_rr.course_ids = ["202611_CSCE_638_500"]
    mock_rr.rewritten_query = "grading"
    mock_rr.requires_retrieval = True

    chunks = [{"content": "Grading: A=90+" * 10}]

    with patch("evals.eval_chunking.classify_query", return_value=mock_rr), \
         patch("evals.eval_chunking.retrieve_for_query", return_value=chunks), \
         patch("evals.eval_chunking.compute_embedding_metrics",
               return_value={"precision_at_k": 1.0, "hit_rate_at_k": 1.0, "retrieved_tokens": 30}):
        from evals.eval_chunking import run_eval
        results, run_name = run_eval(
            [_make_item()], "test_exp", None, 0.85, ragas_enabled=False, lf=None
        )

    assert len(results) == 1
    assert results[0]["precision_at_k"] == 1.0
    assert results[0]["recall_at_k"] is None
    assert results[0]["f1_at_k"] is None
    assert run_name.startswith("test_exp_")


def test_run_eval_logs_to_langfuse():
    """Calls _log_query_to_langfuse and _log_aggregates_to_langfuse when lf is set."""
    mock_rr = MagicMock()
    mock_rr.function = "hybrid_course"
    mock_rr.course_ids = ["202611_CSCE_638_500"]
    mock_rr.rewritten_query = "grading"
    mock_rr.requires_retrieval = True

    mock_lf = MagicMock()

    with patch("evals.eval_chunking.classify_query", return_value=mock_rr), \
         patch("evals.eval_chunking.retrieve_for_query", return_value=[{"content": "x"}]), \
         patch("evals.eval_chunking.compute_embedding_metrics",
               return_value={"precision_at_k": 0.5, "hit_rate_at_k": 1.0, "retrieved_tokens": 10}), \
         patch("evals.eval_chunking._log_query_to_langfuse") as mock_lq, \
         patch("evals.eval_chunking._log_aggregates_to_langfuse") as mock_la:
        from evals.eval_chunking import run_eval
        run_eval([_make_item()], "exp", None, 0.85, ragas_enabled=False, lf=mock_lf)

    mock_lq.assert_called_once()
    mock_la.assert_called_once()
```

- [ ] **Step 2: Run new tests — expect FAIL**

```bash
cd /workspace && python -m pytest tests/test_eval_chunking.py -k "run_eval" -v 2>&1 | tail -15
```

Expected: `AttributeError` (`run_eval` not defined).

- [ ] **Step 3: Implement eval loop + Langfuse helpers**

Append to `evals/eval_chunking.py`:

```python
# ---------------------------------------------------------------------------
# Langfuse logging
# ---------------------------------------------------------------------------

from rag.tools.langfuse import compute_retrieval_ragas, get_langfuse  # noqa: E402


def _log_query_to_langfuse(lf, row: dict, run_name: str) -> None:
    """Create a Langfuse trace for one query and score all non-None metrics."""
    try:
        trace = lf.trace(
            name="retrieval_eval",
            input={"query": row["query"]},
            tags=["chunking_eval", run_name],
            session_id=run_name,
        )
        for metric in (
            "precision_at_k", "hit_rate_at_k", "retrieved_tokens",
            "recall_at_k", "f1_at_k", "context_precision", "context_recall",
        ):
            value = row.get(metric)
            if value is not None:
                lf.create_score(trace_id=trace.id, name=metric, value=float(value))
    except Exception as e:
        logger.warning(f"Langfuse per-query logging failed: {e}")


def _log_aggregates_to_langfuse(lf, results: list[dict], run_name: str) -> None:
    """Log run-level mean metrics as a summary trace in Langfuse."""
    try:
        aggregates = _compute_aggregates(results)
        trace = lf.trace(
            name="retrieval_eval_aggregate",
            input={"run_name": run_name, "n_queries": len(results)},
            output=aggregates,
            tags=["chunking_eval", run_name, "aggregate"],
            session_id=run_name,
        )
        for metric, value in aggregates.items():
            if isinstance(value, float):
                lf.create_score(trace_id=trace.id, name=metric, value=value)
        lf.flush()
    except Exception as e:
        logger.warning(f"Langfuse aggregate logging failed: {e}")


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(
    golden_items: list[dict],
    experiment: str,
    top_k: Optional[int],
    threshold: float,
    ragas_enabled: bool,
    lf,
) -> tuple[list[dict], str]:
    """Run retrieval eval over all golden items.

    Args:
        golden_items:   List of golden set question dicts.
        experiment:     Experiment name prefix for Langfuse run name.
        top_k:          Override for rerank_k (None → use compute_dynamic_k).
        threshold:      Voyage-3 cosine similarity threshold for relevance.
        ragas_enabled:  If True, run RAGAS ContextPrecision + ContextRecall.
        lf:             Langfuse client (or None to skip logging).

    Returns:
        Tuple of (results list, run_name string).
    """
    run_name = f"{experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results: list[dict] = []

    for i, item in enumerate(golden_items, 1):
        query = item.get("question", item.get("query", ""))
        reference = item.get("reference_answer", "")

        if not query:
            continue
        if ragas_enabled and not reference:
            print(f"  [{i:2d}] SKIP (no reference_answer): {query[:60]}")
            continue

        print(f"  [{i:2d}/{len(golden_items)}] {query[:60]}...")

        try:
            rr = classify_query(query)
        except Exception as e:
            print(f"    Router error: {e}")
            continue

        if not rr.requires_retrieval:
            print(f"    Skip: {rr.function} has no retrieval")
            continue

        try:
            chunks = retrieve_for_query(query, rr, top_k=top_k)
        except Exception as e:
            print(f"    Retrieval error: {e}")
            continue

        emb = compute_embedding_metrics(query, chunks, threshold)

        ragas_scores: dict = {}
        if ragas_enabled and chunks and reference:
            contexts = [c.get("content", "") for c in chunks]
            ragas_scores = compute_retrieval_ragas(
                question=query, contexts=contexts, reference=reference
            )

        recall = ragas_scores.get("context_recall")
        precision = emb["precision_at_k"]
        f1 = _compute_f1(precision, recall) if recall is not None else None

        row = {
            "query": query,
            **emb,
            "recall_at_k": recall,
            "f1_at_k": f1,
            "context_precision": ragas_scores.get("context_precision"),
            "context_recall": recall,
        }
        results.append(row)

        if lf:
            _log_query_to_langfuse(lf, row, run_name)

        recall_str = f"  recall={recall:.3f}" if recall is not None else ""
        print(
            f"    prec={precision:.3f}  hit={emb['hit_rate_at_k']:.0f}"
            f"  tokens={emb['retrieved_tokens']}{recall_str}"
        )

    if lf and results:
        _log_aggregates_to_langfuse(lf, results, run_name)

    return results, run_name
```

- [ ] **Step 4: Run all tests — expect PASS**

```bash
cd /workspace && python -m pytest tests/test_eval_chunking.py -v 2>&1 | tail -25
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /workspace && git add evals/eval_chunking.py tests/test_eval_chunking.py
git commit -m "feat(evals): add run_eval loop and Langfuse logging to eval_chunking"
```

---

## Task 5: Add CLI, `print_summary()`, and JSON output to `evals/eval_chunking.py`

**Files:**
- Modify: `evals/eval_chunking.py` (append `print_summary` + `main`)

- [ ] **Step 1: Append `print_summary()` and `main()` to `evals/eval_chunking.py`**

```python
# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], run_name: str, aggregates: dict) -> None:
    """Print aligned per-query and aggregate metrics to stdout."""
    has_ragas = any(r.get("recall_at_k") is not None for r in results)
    print(f"\n{'='*80}")
    print(f"  RETRIEVAL EVAL: {run_name}  |  {len(results)} queries")
    print(f"{'='*80}")

    header = f"  {'Query':<42} {'Prec':>6} {'Hit':>5} {'Tokens':>7}"
    if has_ragas:
        header += f" {'Recall':>7} {'F1':>6}"
    print(header)
    print(f"  {'-'*78}")

    for r in results:
        recall_str = ""
        if has_ragas:
            rec = r.get("recall_at_k")
            f1 = r.get("f1_at_k")
            recall_str = (
                f" {rec:>7.3f} {f1:>6.3f}"
                if rec is not None and f1 is not None
                else f" {'N/A':>7} {'N/A':>6}"
            )
        print(
            f"  {r['query'][:42]:<42}"
            f" {r['precision_at_k']:>6.3f}"
            f" {r['hit_rate_at_k']:>5.0f}"
            f" {r['retrieved_tokens']:>7d}"
            f"{recall_str}"
        )

    print(f"{'='*80}")
    print("  AGGREGATES:")
    for k, v in aggregates.items():
        print(f"    {k:<30} {v}")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval-only chunking benchmark — compare chunking strategies by retrieval quality"
    )
    parser.add_argument(
        "--golden-set", type=Path, required=True,
        help="Path to golden set JSONL (e.g. tamu_data/evals/golden_sets/golden_*.jsonl)",
    )
    parser.add_argument(
        "--experiment", default="chunking_eval",
        help="Experiment name prefix for Langfuse run (default: chunking_eval)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Override rerank_k; default uses compute_dynamic_k() per query",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Voyage-3 cosine similarity threshold for relevance labels (default: 0.85)",
    )
    parser.add_argument(
        "--ragas", action="store_true",
        help="Enable RAGAS ContextPrecision + ContextRecall (costs LLM tokens, ~30s/query)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Write JSON results to this path (optional)",
    )
    args = parser.parse_args()

    if not args.golden_set.exists():
        print(f"ERROR: Golden set not found: {args.golden_set}")
        sys.exit(1)

    print(f"\nLoading golden set: {args.golden_set}")
    golden_items = load_golden_set(args.golden_set)
    print(f"  {len(golden_items)} items loaded")

    lf = get_langfuse()
    print(f"  Langfuse: {'connected' if lf else 'not configured (logging skipped)'}")

    print(
        f"\nRunning eval: ragas={'yes' if args.ragas else 'no'}"
        f"  threshold={args.threshold}"
        f"  top_k={args.top_k or 'auto'}\n"
    )

    results, run_name = run_eval(
        golden_items=golden_items,
        experiment=args.experiment,
        top_k=args.top_k,
        threshold=args.threshold,
        ragas_enabled=args.ragas,
        lf=lf,
    )

    if not results:
        print("No results produced. Check golden set and pipeline connectivity.")
        sys.exit(1)

    aggregates = _compute_aggregates(results)
    print_summary(results, run_name, aggregates)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(
                {"run_name": run_name, "aggregates": aggregates, "items": results},
                f, indent=2,
            )
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full test suite — expect PASS**

```bash
cd /workspace && python -m pytest tests/test_eval_chunking.py tests/test_langfuse_retrieval_ragas.py -v 2>&1 | tail -25
```

Expected: all tests pass.

- [ ] **Step 3: Syntax check**

```bash
cd /workspace && python -c "import evals.eval_chunking; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
cd /workspace && git add evals/eval_chunking.py
git commit -m "feat(evals): add CLI, print_summary, and JSON output to eval_chunking"
```

---

## Task 6: Add `eval-chunking` Makefile target + smoke test

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add target to Makefile**

In `Makefile`, add `eval-chunking` to the `.PHONY` line (after `validate-ragas`):

```makefile
.PHONY: run scrape-catalog scrape-classes scrape-simple-syllabus setup-atlas ingest ingest-dept \
        ingest-corpus test typecheck lint format eval-router probe probe-v3 probe-full \
        eval-draft import-draft bench bench-ragas validate-ragas eval-chunking test-v4 probe-v4 \
        sandbox-up sandbox-down sandbox-shell
```

Then append the target (after `validate-ragas`):

```makefile
eval-chunking:
	python evals/eval_chunking.py \
		--golden-set $(GOLDEN) \
		--experiment $(EXP) \
		$(if $(RAGAS),--ragas,) \
		$(if $(TOP_K),--top-k $(TOP_K),) \
		$(if $(OUTPUT),--output $(OUTPUT),)
```

- [ ] **Step 2: Run `make lint` to verify no syntax errors**

```bash
cd /workspace && make lint 2>&1 | tail -10
```

Expected: clean (no errors in `evals/eval_chunking.py`).

- [ ] **Step 3: Dry-run CLI help**

```bash
cd /workspace && python evals/eval_chunking.py --help
```

Expected: prints usage with all flags (`--golden-set`, `--experiment`, `--top-k`, `--threshold`, `--ragas`, `--output`).

- [ ] **Step 4: Commit**

```bash
cd /workspace && git add Makefile
git commit -m "chore: add eval-chunking Makefile target"
```

---

## Verification

After all tasks, run end-to-end smoke test (requires live MongoDB + Voyage AI):

```bash
# Fast smoke test (embedding metrics only, 10 questions)
python evals/eval_chunking.py \
  --golden-set tamu_data/evals/golden_sets/golden_20260313_draft_v1_sample10.jsonl \
  --experiment smoke_$(date +%Y%m%d)

# Check output:
# - Console: per-query prec/hit/tokens + summary table
# - Langfuse: 10 traces tagged smoke_*, each with 3 scores; 1 aggregate trace

# With RAGAS (costs tokens, ~5 min for 10 queries):
python evals/eval_chunking.py \
  --golden-set tamu_data/evals/golden_sets/golden_20260313_draft_v1_sample10.jsonl \
  --experiment smoke_ragas_$(date +%Y%m%d) \
  --ragas

# Check Langfuse UI: session view for run_name shows per-query recall/f1 + aggregate
```

Compare chunking configs (after re-ingesting with different chunk_size):

```bash
CHUNK_SIZE=600 OVERLAP=100 make ingest-corpus
make eval-chunking GOLDEN=tamu_data/evals/golden_sets/golden_20260313_draft_v1.jsonl EXP=chunk_600_ov100

CHUNK_SIZE=300 OVERLAP=50 make ingest-corpus
make eval-chunking GOLDEN=tamu_data/evals/golden_sets/golden_20260313_draft_v1.jsonl EXP=chunk_300_ov50
```

In Langfuse UI: filter by `chunking_eval` tag, compare session averages for `avg_precision_at_k`, `avg_hit_rate_at_k`, `avg_retrieved_tokens`.
