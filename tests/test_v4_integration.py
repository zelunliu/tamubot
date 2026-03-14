"""Integration tests: full graph run with full mock stack for all 3 main paths."""
from unittest.mock import MagicMock
from rag.v4.graph import build_graph
from rag.v4.interfaces import ComponentRegistry
from rag.v4.observability import V4Tracer
from rag.router import RouterResult


def _make_registry(function: str) -> ComponentRegistry:
    router = MagicMock()
    is_recurrent = function == "recurrent"
    rr = RouterResult(
        course_ids=["202611_CSCE_221_500"] if function not in ("out_of_scope", "semantic_general") else [],
        rewritten_query="test query",
        function=function,
        intent_type="ACADEMIC" if function != "out_of_scope" else None,
        recurrent_search=is_recurrent,
    )
    router.classify.return_value = rr

    retriever = MagicMock()
    retriever.fetch_anchor_chunks.return_value = (
        [{"course_id": "202611_CSCE_221_500", "chunk_index": 0, "text": "anchor"}], [], True
    )
    retriever.hybrid_search.return_value = [{"course_id": "202611_CSCE_221_500", "text": "hybrid"}]
    retriever.semantic_search.return_value = [{"course_id": "202611_CSCE_314_500", "text": "sem"}]
    retriever.get_meeting_times.return_value = {}

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, c, top_k, specific_categories=None: c

    generator = MagicMock()
    generator.generate_stream.return_value = iter(["answer"])
    generator.generate_eval_query.return_value = "eval query"

    return ComponentRegistry(router_llm=router, retriever=retriever, reranker=reranker, generator_llm=generator)


def test_hybrid_course_integration():
    registry = _make_registry("hybrid_course")
    tracer = V4Tracer(None)  # no-op tracer
    graph = build_graph(registry, tracer=tracer)
    result = graph.invoke({
        "query": "CSCE 221 grading?", "node_trace": [], "timing_ms": {},
        "conflicted_course_ids": [], "data_gaps": [], "data_integrity": True,
        "anchor_chunks": [], "discovery_chunks": [], "retrieved_chunks": [],
    })
    assert result["function"] == "hybrid_course"
    assert "retrieval" in result["node_trace"]
    assert "generator" in result["node_trace"]
    assert result["answer"]
    # timing_ms populated by middleware
    assert "retrieval_node" in result.get("timing_ms", {}) or len(result.get("timing_ms", {})) > 0


def test_out_of_scope_integration():
    registry = _make_registry("out_of_scope")
    graph = build_graph(registry)
    result = graph.invoke({"query": "hello", "node_trace": [], "timing_ms": {},
        "conflicted_course_ids": [], "data_gaps": [], "data_integrity": True,
        "anchor_chunks": [], "discovery_chunks": [], "retrieved_chunks": []})
    assert result["function"] == "out_of_scope"
    assert result["answer"]
    assert "generator" not in result["node_trace"]


def test_semantic_general_integration():
    registry = _make_registry("semantic_general")
    graph = build_graph(registry)
    result = graph.invoke({
        "query": "what ML courses exist?", "node_trace": [], "timing_ms": {},
        "conflicted_course_ids": [], "data_gaps": [], "data_integrity": True,
        "anchor_chunks": [], "discovery_chunks": [], "retrieved_chunks": [],
    })
    assert result["function"] == "semantic_general"
    assert "retrieval" in result["node_trace"]
    assert result["answer"]


def test_langfuse_span_names_match_node_names():
    """Langfuse span names should be 'v4_{node_name}'."""
    mock_lf = MagicMock()
    mock_lf.span.return_value = MagicMock()
    tracer = V4Tracer(mock_lf)

    state = {"query": "test", "function": "hybrid_course", "course_ids": [], "node_trace": []}
    with tracer.node_span("retrieval", state):
        pass

    mock_lf.span.assert_called()
    assert mock_lf.span.call_args.kwargs["name"] == "v4_retrieval"
