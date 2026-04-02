"""Tests for router_node behavior: OOS derivation, JSON parse failure fallback."""
from unittest.mock import MagicMock
from rag.v4.nodes.router_node import router_node
from rag.router import RouterResult


def _make_registry(function="hybrid_course"):
    registry = MagicMock()
    rr = RouterResult(
        course_ids=["202611_CSCE_221_500"] if function not in ("out_of_scope", "semantic_general") else [],
        rewritten_query="test query",
        function=function,
        intent_type="ACADEMIC" if function != "out_of_scope" else None,
    )
    registry.router_llm.classify.return_value = rr
    return registry


def test_router_node_hybrid_course():
    registry = _make_registry("hybrid_course")
    result = router_node({"query": "CSCE 221 office hours?", "node_trace": [], "timing_ms": {}}, registry=registry)
    assert result["function"] == "hybrid_course"
    assert result["course_ids"] == ["202611_CSCE_221_500"]
    assert "router" in result["node_trace"]


def test_router_node_out_of_scope_derivation():
    """When course_ids is empty and intent_type is None, function should be out_of_scope."""
    registry = _make_registry("out_of_scope")
    result = router_node({"query": "what's the weather?", "node_trace": [], "timing_ms": {}}, registry=registry)
    assert result["function"] == "out_of_scope"
    assert result["course_ids"] == []


def test_router_node_fallback_to_out_of_scope_on_exception():
    """If classify() throws, router_node falls back to out_of_scope."""
    registry = MagicMock()
    registry.router_llm.classify.side_effect = Exception("LLM timeout")

    result = router_node({"query": "test", "node_trace": [], "timing_ms": {}}, registry=registry)
    assert result["function"] == "out_of_scope"
    assert "error" in result
    assert "Router failed" in result["error"]


def test_router_node_appends_to_node_trace():
    registry = _make_registry("hybrid_course")
    result = router_node({"query": "test", "node_trace": ["prev"], "timing_ms": {}}, registry=registry)
    assert result["node_trace"] == ["prev", "router"]


def test_router_node_passes_prior_context_when_history_present():
    """router_node builds prior_context from history and passes it to classify."""
    from unittest.mock import MagicMock
    from rag.v4.nodes.router_node import router_node
    from rag.router import RouterResult

    registry = MagicMock()
    registry.router_llm.classify.return_value = RouterResult(
        course_ids=["CSCE 638", "CSCE 670"],
        rewritten_query="compare schedule CSCE 638 CSCE 670",
    )

    state = {
        "query": "compare it with CSCE 670",
        "history": [
            {"role": "user", "content": "what's the schedule for CSCE 638?"},
            {
                "role": "assistant",
                "content": "The schedule is MWF 9-10am.",
                "router_result": {
                    "function": "hybrid_course",
                    "course_ids": ["CSCE 638"],
                },
            },
        ],
        "node_trace": [],
        "timing_ms": {},
    }

    router_node(state, registry=registry)

    call_kwargs = registry.router_llm.classify.call_args.kwargs
    assert "prior_context" in call_kwargs
    ctx = call_kwargs["prior_context"]
    assert ctx is not None
    assert "CSCE 638" in ctx


def test_router_node_no_prior_context_when_history_empty():
    """router_node passes prior_context=None when history is empty."""
    from unittest.mock import MagicMock
    from rag.v4.nodes.router_node import router_node
    from rag.router import RouterResult

    registry = MagicMock()
    registry.router_llm.classify.return_value = RouterResult(
        course_ids=[], rewritten_query="test"
    )

    state = {"query": "hello", "history": [], "node_trace": [], "timing_ms": {}}
    router_node(state, registry=registry)

    call_kwargs = registry.router_llm.classify.call_args.kwargs
    assert call_kwargs.get("prior_context") is None
