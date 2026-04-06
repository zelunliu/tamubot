"""Tests for run_pipeline return_timing parameter."""
from unittest.mock import MagicMock, patch


def _make_mock_result(function: str = "hybrid_course") -> dict:
    """Minimal LangGraph result dict with timing_ms populated."""
    return {
        "retrieved_chunks": [{"content": "chunk"}],
        "function": function,
        "course_ids": ["202611_CSCE_221_500"],
        "rewritten_query": "test",
        "intent_type": "ACADEMIC",
        "recursive_search": False,
        "retrieval_mode": "",
        "data_gaps": [],
        "data_integrity": True,
        "timing_ms": {
            "router_node": 12.3,
            "retrieval_node": 45.6,
            "generator_node": 78.9,
        },
    }


def test_run_pipeline_default_returns_5tuple():
    """Default call (no return_timing) still returns 5-tuple."""
    mock_result = _make_mock_result()
    with patch("rag.graph.pipeline._get_graph") as mock_get_graph:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_result
        mock_get_graph.return_value = mock_graph

        from rag.graph.pipeline import run_pipeline
        result = run_pipeline("test query")

    assert len(result) == 5


def test_run_pipeline_return_timing_true_returns_6tuple():
    """return_timing=True appends timing_ms dict as 6th element."""
    mock_result = _make_mock_result()
    with patch("rag.graph.pipeline._get_graph") as mock_get_graph:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_result
        mock_get_graph.return_value = mock_graph

        from rag.graph.pipeline import run_pipeline
        result = run_pipeline("test query", return_timing=True)

    assert len(result) == 6
    timing = result[5]
    assert isinstance(timing, dict)
    assert "router_node" in timing
    assert timing["router_node"] == 12.3


def test_run_pipeline_return_timing_empty_dict_on_missing():
    """If timing_ms is absent from result, returns empty dict (not KeyError)."""
    mock_result = _make_mock_result()
    mock_result.pop("timing_ms")
    with patch("rag.graph.pipeline._get_graph") as mock_get_graph:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_result
        mock_get_graph.return_value = mock_graph

        from rag.graph.pipeline import run_pipeline
        result = run_pipeline("test query", return_timing=True)

    assert result[5] == {}
