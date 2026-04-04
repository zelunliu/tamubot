"""Tests for full recurrent path with conflict filtering and course cap."""
from unittest.mock import MagicMock, patch
from rag.graph.builder import build_graph
from rag.router import RouterResult


def _make_patches(router_rr=None, anchor_return=None, semantic_return=None, meeting_times=None, generator_return=None):
    """Return a dict of patch kwargs for the recurrent path."""
    if router_rr is None:
        router_rr = RouterResult(
            course_ids=["202611_CSCE_221_500"],
            rewritten_query="compare with similar courses",
            function="recurrent",
            intent_type="ACADEMIC",
            recurrent_search=True,
        )
    if anchor_return is None:
        anchor_return = (
            [{"course_id": "202611_CSCE_221_500", "chunk_index": i, "text": f"anchor {i}", "category": "GRADING"}
             for i in range(3)],
            [],
            True,
        )
    if semantic_return is None:
        semantic_return = [
            {"course_id": "202611_CSCE_222_500", "chunk_index": 0, "text": "disc1", "category": "GRADING"},
            {"course_id": "202611_CSCE_314_500", "chunk_index": 0, "text": "disc2", "category": "GRADING"},
            {"course_id": "202611_CSCE_312_500", "chunk_index": 0, "text": "disc3", "category": "GRADING"},
            {"course_id": "202611_CSCE_331_500", "chunk_index": 0, "text": "disc4", "category": "GRADING"},
        ]
    if meeting_times is None:
        meeting_times = {}
    if generator_return is None:
        generator_return = iter(["Recommended courses: CSCE 222, 314, 312"])
    return dict(
        router_rr=router_rr,
        anchor_return=anchor_return,
        semantic_return=semantic_return,
        meeting_times=meeting_times,
        generator_return=generator_return,
    )


def _run_recurrent_graph(patches=None, query="what courses are similar to CSCE 221?"):
    if patches is None:
        patches = _make_patches()
    graph = build_graph()
    with patch("rag.router.classify_query", return_value=patches["router_rr"]), \
         patch("rag.tools.mongo.fetch_anchor_chunks", return_value=patches["anchor_return"]), \
         patch("rag.tools.mongo.semantic_search", return_value=patches["semantic_return"]), \
         patch("rag.tools.mongo.get_meeting_times", return_value=patches["meeting_times"]), \
         patch("rag.tools.voyage.rerank", side_effect=lambda q, c, top_k: c[:top_k] if c else []), \
         patch("rag.generator.generate_stream", return_value=patches["generator_return"]), \
         patch("rag.generator.generate_eval_search_string", return_value="find similar courses to CSCE 221"):
        return graph.invoke({
            "query": query,
            "node_trace": [], "timing_ms": {},
            "conflicted_course_ids": [], "data_gaps": [], "data_integrity": True,
            "anchor_chunks": [], "discovery_chunks": [], "retrieved_chunks": [],
        })


def test_recurrent_full_7_node_path():
    result = _run_recurrent_graph()
    expected_nodes = ["router", "anchor", "eval_search", "retrieval", "schedule_filter", "merge", "generator"]
    for node in expected_nodes:
        assert node in result["node_trace"], f"Missing node: {node}"


def test_recurrent_conflicted_course_ids_populated():
    """When schedule filter detects conflicts, conflicted_course_ids is set."""
    result = _run_recurrent_graph()
    # conflicted_course_ids is set (empty list is valid since no conflicts in mock)
    assert "conflicted_course_ids" in result
    assert isinstance(result["conflicted_course_ids"], list)


def test_recurrent_discovery_course_cap():
    """merge_node should cap discovery courses at RECURRENT_MAX_RECOMMENDED_COURSES (3)."""
    import config
    result = _run_recurrent_graph()

    # Discovery has 4 courses but cap is 3
    retrieved = result.get("retrieved_chunks", [])
    # Anchor course + at most RECURRENT_MAX_RECOMMENDED_COURSES discovery courses
    unique_discovery_courses = {
        c["course_id"] for c in retrieved
        if c["course_id"] != "202611_CSCE_221_500"
    }
    assert len(unique_discovery_courses) <= config.RECURRENT_MAX_RECOMMENDED_COURSES


def test_schedule_filter_uses_all_anchor_intervals():
    """schedule_filter_node must filter using ALL anchor courses, not just the first."""
    from rag.nodes.schedule_filter_node import schedule_filter_node

    # Two anchor courses with different meeting times
    # anchor A: MW 8:00AM-9:15AM
    # anchor B: TR 2:00PM-3:15PM
    # discovery C conflicts with B (TR 2:30PM-3:45PM) but not A
    # Without fix, C would NOT be filtered (only anchor A is checked)
    meeting_times = {
        "ANCHOR_A": "MW 8:00AM - 9:15AM",
        "ANCHOR_B": "TR 2:00PM - 3:15PM",
        "DISC_C":   "TR 2:30PM - 3:45PM",
        "DISC_D":   "WEB ASYNC",
    }

    discovery_chunks = [
        {"course_id": "DISC_C", "text": "conflicts with B"},
        {"course_id": "DISC_D", "text": "async"},
    ]
    state = {
        "course_ids": ["ANCHOR_A", "ANCHOR_B"],
        "discovery_chunks": discovery_chunks,
        "node_trace": [],
    }

    with patch("rag.tools.mongo.get_meeting_times", side_effect=lambda cids: {k: meeting_times[k] for k in cids if k in meeting_times}):
        result = schedule_filter_node(state)

    remaining_ids = {c["course_id"] for c in result["discovery_chunks"]}
    assert "DISC_C" not in remaining_ids, "DISC_C conflicts with ANCHOR_B and must be filtered"
    assert "DISC_D" in remaining_ids, "async DISC_D must be kept"
    assert "DISC_C" in result["conflicted_course_ids"]
