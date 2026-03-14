"""Tests for full recurrent path with conflict filtering and course cap."""
from unittest.mock import MagicMock
from rag.v4.graph import build_graph
from rag.v4.interfaces import ComponentRegistry
from rag.router import RouterResult


def _make_recurrent_registry():
    router = MagicMock()
    rr = RouterResult(
        course_ids=["202611_CSCE_221_500"],
        rewritten_query="compare with similar courses",
        function="recurrent",
        intent_type="ACADEMIC",
        recurrent_search=True,
    )
    router.classify.return_value = rr

    retriever = MagicMock()
    retriever.fetch_anchor_chunks.return_value = (
        [{"course_id": "202611_CSCE_221_500", "chunk_index": i, "text": f"anchor {i}", "category": "GRADING"}
         for i in range(3)],
        [],
        True,
    )
    retriever.semantic_search.return_value = [
        {"course_id": "202611_CSCE_222_500", "chunk_index": 0, "text": "disc1", "category": "GRADING"},
        {"course_id": "202611_CSCE_314_500", "chunk_index": 0, "text": "disc2", "category": "GRADING"},
        {"course_id": "202611_CSCE_312_500", "chunk_index": 0, "text": "disc3", "category": "GRADING"},
        {"course_id": "202611_CSCE_331_500", "chunk_index": 0, "text": "disc4", "category": "GRADING"},
    ]
    # Return empty meeting times → no schedule conflicts
    retriever.get_meeting_times.return_value = {}

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, chunks, top_k, specific_categories=None: chunks

    generator = MagicMock()
    generator.generate_stream.return_value = iter(["Recommended courses: CSCE 222, 314, 312"])
    generator.generate_eval_query.return_value = "find similar courses to CSCE 221"

    return ComponentRegistry(
        router_llm=router,
        retriever=retriever,
        reranker=reranker,
        generator_llm=generator,
    )


def test_recurrent_full_7_node_path():
    registry = _make_recurrent_registry()
    graph = build_graph(registry)
    result = graph.invoke({
        "query": "what courses are similar to CSCE 221?",
        "node_trace": [], "timing_ms": {},
        "conflicted_course_ids": [], "data_gaps": [], "data_integrity": True,
        "anchor_chunks": [], "discovery_chunks": [], "retrieved_chunks": [],
    })

    expected_nodes = ["router", "anchor", "eval_search", "retrieval", "schedule_filter", "merge", "generator"]
    for node in expected_nodes:
        assert node in result["node_trace"], f"Missing node: {node}"


def test_recurrent_conflicted_course_ids_populated():
    """When schedule filter detects conflicts, conflicted_course_ids is set."""
    registry = _make_recurrent_registry()
    graph = build_graph(registry)
    result = graph.invoke({
        "query": "compare CSCE 221", "node_trace": [], "timing_ms": {},
        "conflicted_course_ids": [], "data_gaps": [], "data_integrity": True,
        "anchor_chunks": [], "discovery_chunks": [], "retrieved_chunks": [],
    })
    # conflicted_course_ids is set (empty list is valid since no conflicts in mock)
    assert "conflicted_course_ids" in result
    assert isinstance(result["conflicted_course_ids"], list)


def test_recurrent_discovery_course_cap():
    """merge_node should cap discovery courses at RECURRENT_MAX_RECOMMENDED_COURSES (3)."""
    import config
    registry = _make_recurrent_registry()
    graph = build_graph(registry)
    result = graph.invoke({
        "query": "compare CSCE 221", "node_trace": [], "timing_ms": {},
        "conflicted_course_ids": [], "data_gaps": [], "data_integrity": True,
        "anchor_chunks": [], "discovery_chunks": [], "retrieved_chunks": [],
    })

    # Discovery has 4 courses but cap is 3
    retrieved = result.get("retrieved_chunks", [])
    # Anchor course + at most RECURRENT_MAX_RECOMMENDED_COURSES discovery courses
    unique_discovery_courses = {
        c["course_id"] for c in retrieved
        if c["course_id"] != "202611_CSCE_221_500"
    }
    assert len(unique_discovery_courses) <= config.RECURRENT_MAX_RECOMMENDED_COURSES
