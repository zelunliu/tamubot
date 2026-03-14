"""Tests for the v4 LangGraph graph with mock registry."""
from unittest.mock import MagicMock
from rag.v4.graph import build_graph
from rag.v4.interfaces import ComponentRegistry
from rag.router import RouterResult


def _make_mock_registry(function: str = "hybrid_course") -> ComponentRegistry:
    """Build a mock registry returning canned RouterResult."""
    router = MagicMock()
    rr = RouterResult(
        course_ids=["202611_CSCE_221_500"] if function != "semantic_general" else [],
        rewritten_query="test query",
        function=function,
        intent_type="ACADEMIC" if function != "out_of_scope" else None,
    )
    router.classify.return_value = rr

    retriever = MagicMock()
    retriever.fetch_anchor_chunks.return_value = (
        [{"course_id": "202611_CSCE_221_500", "chunk_index": 0, "text": "anchor chunk"}],
        [],
        True,
    )
    retriever.hybrid_search.return_value = [{"course_id": "202611_CSCE_221_500", "chunk_index": 1, "text": "result"}]
    retriever.semantic_search.return_value = [{"course_id": "202611_CSCE_314_500", "chunk_index": 0, "text": "sem result"}]
    retriever.get_meeting_times.return_value = {}

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, chunks, top_k: chunks[:top_k] if chunks else []

    generator = MagicMock()
    generator.generate_stream.return_value = iter(["Hello ", "world"])
    generator.generate_eval_query.return_value = "expanded query"

    return ComponentRegistry(
        router_llm=router,
        retriever=retriever,
        reranker=reranker,
        generator_llm=generator,
    )


def test_hybrid_course_path():
    registry = _make_mock_registry("hybrid_course")
    graph = build_graph(registry)
    result = graph.invoke({"query": "what are office hours?", "node_trace": [], "timing_ms": {}})

    assert "hybrid_course" in result["node_trace"] or "retrieval" in result["node_trace"]
    assert "generator" in result["node_trace"]
    assert result["function"] == "hybrid_course"
    assert isinstance(result["answer"], str)


def test_out_of_scope_path():
    registry = _make_mock_registry("out_of_scope")
    graph = build_graph(registry)
    result = graph.invoke({"query": "what is the weather?", "node_trace": [], "timing_ms": {}})

    assert "out_of_scope" in result["node_trace"]
    assert "generator" not in result["node_trace"]
    assert result["answer"]  # canned response


def test_semantic_general_path():
    registry = _make_mock_registry("semantic_general")
    graph = build_graph(registry)
    result = graph.invoke({"query": "what CSCE courses cover ML?", "node_trace": [], "timing_ms": {}})

    assert "retrieval" in result["node_trace"]
    assert "generator" in result["node_trace"]
    assert result["function"] == "semantic_general"


def test_recurrent_path():
    registry = _make_mock_registry("recurrent")
    # Need recurrent=True on RouterResult
    rr = RouterResult(
        course_ids=["202611_CSCE_221_500"],
        rewritten_query="compare with similar courses",
        function="recurrent",
        intent_type="ACADEMIC",
        recurrent_search=True,
    )
    registry.router_llm.classify.return_value = rr

    graph = build_graph(registry)
    result = graph.invoke({
        "query": "compare CSCE 221 with similar courses",
        "node_trace": [],
        "timing_ms": {},
        "conflicted_course_ids": [],
        "data_gaps": [],
        "data_integrity": True,
        "anchor_chunks": [],
        "discovery_chunks": [],
        "retrieved_chunks": [],
    })

    assert "anchor" in result["node_trace"]
    assert "eval_search" in result["node_trace"]
    assert "retrieval" in result["node_trace"]
    assert "schedule_filter" in result["node_trace"]
    assert "merge" in result["node_trace"]
    assert "generator" in result["node_trace"]


def test_result_is_v3_compatible():
    """The graph result must contain all 6 fields returned by v3 run_pipeline()."""
    registry = _make_mock_registry("hybrid_course")
    graph = build_graph(registry)
    result = graph.invoke({
        "query": "test",
        "node_trace": [],
        "timing_ms": {},
        "conflicted_course_ids": [],
        "data_gaps": [],
        "data_integrity": True,
        "anchor_chunks": [],
        "discovery_chunks": [],
        "retrieved_chunks": [],
    })

    # v3 returns: (chunks, router_result, data_gaps, data_integrity, conflicted_ids, timing_ms)
    assert "retrieved_chunks" in result
    assert "router_result" in result
    assert "data_gaps" in result
    assert "data_integrity" in result
    assert "conflicted_course_ids" in result
