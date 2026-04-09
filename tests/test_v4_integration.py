"""Integration tests: full graph run with mock tools for all 3 main paths."""
from unittest.mock import patch

from rag.graph.builder import build_graph
from rag.router import RouterResult


def _invoke_graph(function: str):
    is_recursive = function == "recursive"
    rr = RouterResult(
        course_ids=["202611_CSCE_221_500"] if function not in ("out_of_scope", "semantic_general") else [],
        rewritten_query="test query",
        function=function,
        intent_type="ACADEMIC" if function != "out_of_scope" else None,
        recursive_search=is_recursive,
    )

    with patch("rag.router.classify_query", return_value=rr), \
         patch("rag.tools.mongo.hybrid_search", return_value=[
             {"course_id": "202611_CSCE_221_500", "text": "hybrid"}
         ]), \
         patch("rag.tools.mongo.semantic_search", return_value=[
             {"course_id": "202611_CSCE_314_500", "text": "sem"}
         ]), \
         patch("rag.tools.voyage.rerank", side_effect=lambda q, c, top_k: c), \
         patch("rag.generator.generate_stream", return_value=iter(["answer"])), \
         patch("rag.tools.llm.call_llm") as mock_llm:
        mock_llm.return_value.text = (
            '{"function": "semantic_general", "course_ids": [], "rewritten_query": "related courses"}'
        )
        graph = build_graph()
        return graph.invoke({
            "query": "test", "node_trace": [], "timing_ms": {},
            "data_gaps": [], "data_integrity": True,
            "recursive_chunks": [], "retrieved_chunks": [],
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


def test_callbackhandler_importable():
    """CallbackHandler should be importable from langfuse.langchain."""
    from langfuse.langchain import CallbackHandler
    assert CallbackHandler is not None
