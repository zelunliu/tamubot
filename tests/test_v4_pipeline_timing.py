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
