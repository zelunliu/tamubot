"""Unit tests for compute_retrieval_ragas() in rag/tools/langfuse.py."""
import math
from unittest.mock import MagicMock, patch

import pandas as pd


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


def test_compute_retrieval_ragas_returns_only_numeric_scores():
    """Returns filtered dict (numeric, non-NaN only) — excludes string fields like user_input."""
    mock_df = pd.DataFrame([{
        "context_precision": 0.75,
        "context_recall": 0.8,
        "user_input": "What is grading?",
    }])
    mock_eval_result = MagicMock()
    mock_eval_result.to_pandas.return_value = mock_df
    mock_metric = MagicMock()

    with patch("rag.tools.langfuse.get_langfuse", return_value=None), \
         patch("langchain_openai.ChatOpenAI"), \
         patch("ragas.llms.LangchainLLMWrapper"), \
         patch("ragas.SingleTurnSample"), \
         patch("ragas.EvaluationDataset"), \
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
    assert "user_input" not in result


def test_compute_retrieval_ragas_uploads_scores_to_langfuse():
    """Calls lf.create_score() for each numeric metric when trace_id is provided."""
    mock_df = pd.DataFrame([{"context_precision": 0.6, "context_recall": 0.7}])
    mock_eval_result = MagicMock()
    mock_eval_result.to_pandas.return_value = mock_df
    mock_lf = MagicMock()
    mock_metric = MagicMock()

    with patch("rag.tools.langfuse.get_langfuse", return_value=mock_lf), \
         patch("langchain_openai.ChatOpenAI"), \
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
