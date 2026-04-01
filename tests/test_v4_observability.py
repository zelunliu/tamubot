"""Tests for V4Tracer and middleware."""
import time
from unittest.mock import MagicMock

from rag.v4.observability import V4Tracer
from rag.v4.middleware import timing_middleware, error_guard_middleware
from rag.v4.exceptions import V4PipelineError


# ── V4Tracer tests ──────────────────────────────────────────────────────────

def test_v4_tracer_none_is_noop():
    """V4Tracer(None) must not crash — all methods are no-ops."""
    tracer = V4Tracer(None)
    state = {"query": "test", "function": "hybrid_course", "course_ids": [], "node_trace": []}
    with tracer.node_span("router", state) as span:
        assert span is None  # no-op returns None

    # trace_node_transition is also a no-op
    tracer.trace_node_transition("router", "retrieval", state)


def test_v4_tracer_with_mock_client_calls_span():
    """With a mock lf_client, V4Tracer.node_span opens and closes a span."""
    mock_lf = MagicMock()
    mock_span = MagicMock()
    mock_lf.span.return_value = mock_span

    tracer = V4Tracer(mock_lf)
    state = {"query": "test", "function": "hybrid_course", "course_ids": ["A"], "node_trace": []}

    with tracer.node_span("router", state) as span:
        assert span is mock_span

    mock_lf.span.assert_called_once()
    call_kwargs = mock_lf.span.call_args
    assert call_kwargs.kwargs["name"] == "v4_router"
    mock_span.end.assert_called_once()


def test_v4_tracer_span_names_match_node_names():
    """Span name should be 'v4_{node_name}'."""
    mock_lf = MagicMock()
    mock_lf.span.return_value = MagicMock()
    tracer = V4Tracer(mock_lf)
    state = {"query": "x", "function": "recurrent", "course_ids": [], "node_trace": []}

    with tracer.node_span("anchor", state):
        pass

    assert mock_lf.span.call_args.kwargs["name"] == "v4_anchor"


# ── timing_middleware tests ──────────────────────────────────────────────────

def test_timing_middleware_populates_timing_ms():
    """timing_middleware must add node_name key to state['timing_ms']."""
    @timing_middleware
    def mock_node(state, registry=None):
        return {"answer": "hello"}

    result = mock_node({"timing_ms": {}}, registry=None)
    assert "mock_node" in result["timing_ms"]
    assert isinstance(result["timing_ms"]["mock_node"], float)
    assert result["timing_ms"]["mock_node"] >= 0.0


def test_timing_middleware_merges_existing_timing():
    """timing_middleware must preserve existing timing_ms entries."""
    @timing_middleware
    def another_node(state, registry=None):
        return {}

    result = another_node({"timing_ms": {"prev_node": 42.0}}, registry=None)
    assert "prev_node" in result["timing_ms"]
    assert "another_node" in result["timing_ms"]


# ── error_guard_middleware tests ─────────────────────────────────────────────

def test_error_guard_catches_v4_pipeline_error():
    """error_guard_middleware catches V4PipelineError and writes state['error']."""
    @error_guard_middleware
    def failing_node(state, registry=None):
        raise V4PipelineError("retrieval timeout")

    result = failing_node({}, registry=None)
    assert "error" in result
    assert "retrieval timeout" in result["error"]


def test_error_guard_does_not_catch_non_v4_errors():
    """error_guard_middleware re-raises non-V4PipelineError exceptions."""
    @error_guard_middleware
    def buggy_node(state, registry=None):
        raise ValueError("unexpected bug")

    try:
        buggy_node({}, registry=None)
        assert False, "Should have raised"
    except ValueError:
        pass  # expected


def test_error_guard_passes_through_successful_result():
    """error_guard_middleware returns result unchanged on success."""
    @error_guard_middleware
    def good_node(state, registry=None):
        return {"answer": "ok", "node_trace": ["good_node"]}

    result = good_node({})
    assert result["answer"] == "ok"


def test_trace_registry_register_and_get():
    from rag.v4.trace_registry import register, get, clear
    mock_trace = object()
    register("session-abc", mock_trace)
    assert get("session-abc") is mock_trace
    clear("session-abc")


def test_trace_registry_clear_removes_entry():
    from rag.v4.trace_registry import register, get, clear
    register("session-xyz", object())
    clear("session-xyz")
    assert get("session-xyz") is None


def test_trace_registry_empty_session_id_is_noop():
    from rag.v4.trace_registry import register, get
    register("", object())
    assert get("") is None


def test_trace_registry_unknown_session_returns_none():
    from rag.v4.trace_registry import get
    assert get("never-registered-session-99") is None


def test_pipeline_registers_trace_before_invoke():
    """run_pipeline_v4_with_memory registers trace in registry before invoking the graph."""
    from unittest.mock import MagicMock, patch
    import rag.v4.trace_registry as reg

    mock_trace = MagicMock()
    registered = {}

    original_register = reg.register
    def spy_register(session_id, trace):
        registered[session_id] = trace
        original_register(session_id, trace)

    mock_graph_result = {
        "retrieved_chunks": [],
        "router_result": MagicMock(function="out_of_scope", course_ids=[], requires_retrieval=False),
        "data_gaps": [],
        "data_integrity": True,
        "conflicted_course_ids": [],
        "answer_stream": [],
        "function": "out_of_scope",
    }

    with patch("rag.v4.pipeline_v4._memory_graph") as mock_graph, \
         patch.object(reg, "register", side_effect=spy_register):
        mock_graph.invoke.return_value = mock_graph_result
        from rag.v4.pipeline_v4 import run_pipeline_v4_with_memory
        run_pipeline_v4_with_memory(
            "hello",
            trace=mock_trace,
            thread_config={"configurable": {"thread_id": "t-trace-test"}},
        )

    assert "t-trace-test" in registered
    assert registered["t-trace-test"] is mock_trace
