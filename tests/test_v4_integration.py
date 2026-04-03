"""Integration tests: full graph run with mock tools for all 3 main paths."""
from unittest.mock import MagicMock, patch
from rag.graph.builder import build_graph
from rag.router import RouterResult


def _invoke_graph(function: str):
    is_recurrent = function == "recurrent"
    rr = RouterResult(
        course_ids=["202611_CSCE_221_500"] if function not in ("out_of_scope", "semantic_general") else [],
        rewritten_query="test query",
        function=function,
        intent_type="ACADEMIC" if function != "out_of_scope" else None,
        recurrent_search=is_recurrent,
    )

    with patch("rag.router.classify_query", return_value=rr), \
         patch("rag.tools.mongo.fetch_anchor_chunks", return_value=(
             [{"course_id": "202611_CSCE_221_500", "chunk_index": 0, "text": "anchor"}], [], True
         )), \
         patch("rag.tools.mongo.hybrid_search", return_value=[
             {"course_id": "202611_CSCE_221_500", "text": "hybrid"}
         ]), \
         patch("rag.tools.mongo.semantic_search", return_value=[
             {"course_id": "202611_CSCE_314_500", "text": "sem"}
         ]), \
         patch("rag.tools.mongo.get_meeting_times", return_value={}), \
         patch("rag.tools.voyage.rerank", side_effect=lambda q, c, top_k: c), \
         patch("rag.generator.generate_stream", return_value=iter(["answer"])), \
         patch("rag.generator.generate_eval_search_string", return_value="eval query"):
        graph = build_graph()
        return graph.invoke({
            "query": "test", "node_trace": [], "timing_ms": {},
            "conflicted_course_ids": [], "data_gaps": [], "data_integrity": True,
            "anchor_chunks": [], "discovery_chunks": [], "retrieved_chunks": [],
        })


def test_hybrid_course_integration():
    result = _invoke_graph("hybrid_course")
    assert result["function"] == "hybrid_course"
    assert "retrieval" in result["node_trace"]
    assert "generator" in result["node_trace"]
    assert result["answer"]
    assert len(result.get("timing_ms", {})) > 0


def test_out_of_scope_integration():
    result = _invoke_graph("out_of_scope")
    assert result["function"] == "out_of_scope"
    assert result["answer"]
    assert "generator" not in result["node_trace"]


def test_semantic_general_integration():
    result = _invoke_graph("semantic_general")
    assert result["function"] == "semantic_general"
    assert "retrieval" in result["node_trace"]
    assert result["answer"]


def test_langfuse_span_names_match_node_names():
    """Langfuse span names should be 'v4_{node_name}' — tested via V4Tracer in v4.observability."""
    from rag.tools.langfuse import V4Tracer

    mock_lf = MagicMock()
    mock_lf.span.return_value = MagicMock()
    tracer = V4Tracer(mock_lf)

    state = {"query": "test", "function": "hybrid_course", "course_ids": [], "node_trace": []}
    with tracer.node_span("retrieval", state):
        pass

    mock_lf.span.assert_called()
    assert mock_lf.span.call_args.kwargs["name"] == "v4_retrieval"
