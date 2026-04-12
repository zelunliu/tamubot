"""Unit tests for pure functions in evals/eval_chunking.py."""
import json
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# load_golden_set
# ---------------------------------------------------------------------------

def test_golden_set_load_returns_all_items(tmp_path):
    """Reads every non-empty row from an xlsx golden set."""
    from evals.golden_set import save, load
    data = [
        {"id": 1, "question": "What is grading?", "reference_answer": "A/B/C",
         "expected_function": "hybrid_course", "human_notes": None},
        {"id": 2, "question": "Who teaches it?", "reference_answer": "Dr. Smith",
         "expected_function": "hybrid_course", "human_notes": None},
    ]
    path = tmp_path / "golden.xlsx"
    save(data, path)
    result = load(path)
    assert len(result) == 2
    assert result[0]["question"] == "What is grading?"


def test_golden_set_load_skips_empty_question_rows(tmp_path):
    """Ignores rows with empty question in xlsx golden set."""
    from evals.golden_set import save, load
    data = [
        {"id": 1, "question": "Q1", "reference_answer": "A1",
         "expected_function": "hybrid_course", "human_notes": None},
        {"id": 2, "question": "", "reference_answer": "",
         "expected_function": "", "human_notes": None},
    ]
    path = tmp_path / "golden.xlsx"
    save(data, path)
    result = load(path)
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


# ---------------------------------------------------------------------------
# retrieve_for_query
# ---------------------------------------------------------------------------

def _make_mock_router_result(function="hybrid_course", course_ids=None, requires_retrieval=True):
    mock_rr = MagicMock()
    mock_rr.function = function
    mock_rr.course_ids = course_ids or ["202611_CSCE_638_500"]
    mock_rr.requires_retrieval = requires_retrieval
    mock_rr.rewritten_query = "query"
    return mock_rr


def test_run_one_query_skips_out_of_scope():
    """Returns None when pipeline reports no retrieval required."""
    mock_rr = _make_mock_router_result(function="out_of_scope", course_ids=[], requires_retrieval=False)

    with patch("evals.eval_chunking.run_pipeline_eval",
               return_value=([], mock_rr, {})):
        from evals.eval_chunking import _run_one_query
        result = _run_one_query("query", "ref", None, 0.35, False, 1, 1)

    assert result is None


def test_run_one_query_applies_top_k_slice():
    """top_k slices the chunks returned by the pipeline before metric calculation."""
    mock_rr = _make_mock_router_result()
    chunks = [{"content": f"chunk{i}"} for i in range(10)]

    with patch("evals.eval_chunking.run_pipeline_eval",
               return_value=(chunks, mock_rr, {})), \
         patch("evals.eval_chunking.compute_embedding_metrics",
               return_value={"precision_at_k": 0.5, "hit_rate_at_k": 1.0, "retrieved_tokens": 20}) as mock_emb:
        from evals.eval_chunking import _run_one_query
        _run_one_query("query", "ref", 3, 0.35, False, 1, 1)

    # Only 3 chunks passed to embedding metrics
    called_chunks = mock_emb.call_args[0][1]
    assert len(called_chunks) == 3


# ---------------------------------------------------------------------------
# run_eval (main loop)
# ---------------------------------------------------------------------------

def _make_item(question="What is grading?", reference="Letter grades."):
    return {"question": question, "reference_answer": reference}


def test_run_eval_skips_item_without_reference_when_ragas():
    """Items without reference_answer are skipped when --ragas is on."""
    items = [{"question": "Q1", "reference_answer": ""}]

    with patch("evals.eval_chunking.run_pipeline_eval") as mock_pipe:
        from evals.eval_chunking import run_eval
        results, _, _ = run_eval(items, "test", "test_ds", None, 0.85, ragas_enabled=True, lf=None)

    mock_pipe.assert_not_called()
    assert results == []


def test_run_eval_skips_item_without_question():
    """Items missing 'question' key are skipped."""
    items = [{"reference_answer": "ref"}]

    with patch("evals.eval_chunking.run_pipeline_eval") as mock_pipe:
        from evals.eval_chunking import run_eval
        results, _, _ = run_eval(items, "test", "test_ds", None, 0.85, ragas_enabled=False, lf=None)

    mock_pipe.assert_not_called()
    assert results == []


def test_run_eval_returns_embedding_metrics_only():
    """Without --ragas, returns precision, hit_rate, tokens per query."""
    mock_rr = _make_mock_router_result()
    chunks = [{"content": "Grading: A=90+" * 10}]

    with patch("evals.eval_chunking.run_pipeline_eval", return_value=(chunks, mock_rr, {})), \
         patch("evals.eval_chunking.compute_embedding_metrics",
               return_value={"precision_at_k": 1.0, "hit_rate_at_k": 1.0, "retrieved_tokens": 30}):
        from evals.eval_chunking import run_eval
        results, run_name, _ = run_eval(
            [_make_item()], "test_exp", "test_ds", None, 0.85, ragas_enabled=False, lf=None
        )

    assert len(results) == 1
    assert results[0]["precision_at_k"] == 1.0
    assert results[0]["recall_at_k"] is None
    assert results[0]["f1_at_k"] is None
    assert run_name.startswith("test_exp_")


def test_run_eval_logs_trace_and_scores_to_langfuse():
    """When lf is set: creates one trace per query, scores each metric, links dataset item."""
    mock_rr = _make_mock_router_result()
    mock_lf = MagicMock()
    mock_span = MagicMock()
    mock_span.trace_id = "trace-abc"
    mock_lf.start_observation.return_value = mock_span

    with patch("evals.eval_chunking.upsert_langfuse_dataset") as mock_upsert, \
         patch("evals.eval_chunking.run_pipeline_eval",
               return_value=([{"content": "x"}], mock_rr, {})), \
         patch("evals.eval_chunking.compute_embedding_metrics",
               return_value={"precision_at_k": 0.5, "hit_rate_at_k": 1.0, "retrieved_tokens": 10}):
        from evals.eval_chunking import run_eval
        results, run_name, _ = run_eval(
            [_make_item()], "exp", "test_ds", None, 0.85, ragas_enabled=False, lf=mock_lf
        )

    # dataset upsert called
    mock_upsert.assert_called_once()
    # one span opened per query
    mock_lf.start_observation.assert_called_once()
    # span ended before scoring
    mock_span.end.assert_called_once()
    # scores posted via create_score (not score_trace)
    score_names = {call.kwargs["name"] for call in mock_lf.create_score.call_args_list}
    assert "precision_at_k" in score_names
    assert "hit_rate_at_k" in score_names
    assert "retrieved_tokens" in score_names
    # dataset item linked via dataset_run_items
    mock_lf.api.dataset_run_items.create.assert_called_once()
    # results returned
    assert len(results) == 1
    assert results[0]["precision_at_k"] == 0.5
