"""Generator node — streams the final answer."""
from __future__ import annotations

from rag.graph.middleware import error_guard_middleware, timing_middleware, tracing_middleware
from rag.graph.trace_registry import current_span as _current_span, get as _get_trace
from rag.state.pipeline_state import PipelineState


@tracing_middleware
@timing_middleware
@error_guard_middleware
def generator_node(state: PipelineState) -> dict:
    """Generate the answer. Stores answer_stream as list[str] (picklable for LangGraph)."""
    from rag.generator import generate_stream
    node_trace = list(state.get("node_trace", []))
    node_trace.append("generator")

    # Prefer current node span (set by tracing_middleware), fall back to root trace
    trace = _current_span() or _get_trace(state.get("session_id", "")) or state.get("trace")

    try:
        tokens = list(generate_stream(
            results=state.get("retrieved_chunks", []),
            question=state.get("rewritten_query") or state.get("query", ""),
            function=state.get("function", "semantic_general"),
            course_ids=state.get("course_ids", []),
            intent_type=state.get("intent_type"),
            data_gaps=state.get("data_gaps", []),
            data_integrity=state.get("data_integrity", True),
            conflicted_course_ids=state.get("conflicted_course_ids", []),
            trace=trace,
            history_context=state.get("history_context"),
        ))
        return {
            "answer": "".join(tokens),
            "answer_stream": tokens,
            "node_trace": node_trace,
        }
    except Exception as e:
        err_msg = f"Generation failed: {e}"
        return {
            "answer": err_msg,
            "answer_stream": [err_msg],
            "error": err_msg,
            "node_trace": node_trace,
        }
