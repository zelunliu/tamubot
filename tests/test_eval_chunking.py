"""Unit tests for pure functions in evals/eval_chunking.py."""
import json
from unittest.mock import MagicMock, patch

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
         patch("evals.eval_chunking.semantic_search") as mock_sem, \
         patch("evals.eval_chunking.rerank"):
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

    mock_hs.assert_called_once_with("grading policy", "202611_CSCE_638_500", 20)
    mock_rr_fn.assert_called_once_with("grading policy", fake_chunks, 5)
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

    mock_rr_fn.assert_called_once_with("query", [{"content": "c"}], 3)


def test_retrieve_for_query_semantic_general():
    """Calls semantic_search (not hybrid_search) for semantic_general function."""
    mock_rr = MagicMock()
    mock_rr.function = "semantic_general"
    mock_rr.course_ids = []
    mock_rr.rewritten_query = "related courses"

    with patch("evals.eval_chunking.compute_dynamic_k", return_value={"retrieve_k": 30, "rerank_k": 10}), \
         patch("evals.eval_chunking.semantic_search", return_value=[{"content": "s"}]) as mock_sem, \
         patch("evals.eval_chunking.rerank", return_value=[{"content": "s"}]):
        from evals.eval_chunking import retrieve_for_query
        retrieve_for_query("semantic query", mock_rr, top_k=None)

    mock_sem.assert_called_once_with("related courses", 30)


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
        results, _ = run_eval(items, "test", "test_ds", None, 0.85, ragas_enabled=True, lf=None)

    mock_cq.assert_not_called()
    assert results == []


def test_run_eval_skips_item_without_question():
    """Items missing 'question' key are skipped."""
    items = [{"reference_answer": "ref"}]

    with patch("evals.eval_chunking.classify_query") as mock_cq:
        from evals.eval_chunking import run_eval
        results, _ = run_eval(items, "test", "test_ds", None, 0.85, ragas_enabled=False, lf=None)

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
            [_make_item()], "test_exp", "test_ds", None, 0.85, ragas_enabled=False, lf=None
        )

    assert len(results) == 1
    assert results[0]["precision_at_k"] == 1.0
    assert results[0]["recall_at_k"] is None
    assert results[0]["f1_at_k"] is None
    assert run_name.startswith("test_exp_")


def test_run_eval_logs_trace_and_scores_to_langfuse():
    """When lf is set: creates one trace per query, scores each metric, links dataset item."""
    mock_rr = MagicMock()
    mock_rr.function = "hybrid_course"
    mock_rr.course_ids = ["CSCE_638"]
    mock_rr.rewritten_query = "grading"
    mock_rr.requires_retrieval = True

    mock_lf = MagicMock()
    mock_span = MagicMock()
    mock_span.trace_id = "trace-abc"
    mock_lf.start_observation.return_value = mock_span

    with patch("evals.eval_chunking.upsert_langfuse_dataset") as mock_upsert, \
         patch("evals.eval_chunking.classify_query", return_value=mock_rr), \
         patch("evals.eval_chunking.retrieve_for_query", return_value=[{"content": "x"}]), \
         patch("evals.eval_chunking.compute_embedding_metrics",
               return_value={"precision_at_k": 0.5, "hit_rate_at_k": 1.0, "retrieved_tokens": 10}):
        from evals.eval_chunking import run_eval
        results, run_name = run_eval(
            [_make_item()], "exp", "test_ds", None, 0.85, ragas_enabled=False, lf=mock_lf
        )

    # dataset upsert called
    mock_upsert.assert_called_once()
    # one trace created per query
    mock_lf.start_observation.assert_called_once()
    # scores attached
    score_names = {call.kwargs["name"] for call in mock_span.score_trace.call_args_list}
    assert "precision_at_k" in score_names
    assert "hit_rate_at_k" in score_names
    assert "retrieved_tokens" in score_names
    # dataset item linked via source_trace_id
    mock_lf.create_dataset_item.assert_called_once()
    assert mock_lf.create_dataset_item.call_args.kwargs["source_trace_id"] == "trace-abc"
    # results returned
    assert len(results) == 1
    assert results[0]["precision_at_k"] == 0.5
