"""Out-of-scope node — returns canned response without LLM call."""
from __future__ import annotations

from typing import Any

from rag.v4.middleware import error_guard_middleware, timing_middleware, tracing_middleware
from rag.v4.state import PipelineState

_OOS_RESPONSE = (
    "Howdy! I'm TamuBot, your Texas A&M academic assistant. "
    "I can help you with questions about courses, syllabi, grading policies, "
    "schedules, and university policies. What would you like to know?"
)


@tracing_middleware
@timing_middleware
@error_guard_middleware
def out_of_scope_node(state: PipelineState, registry: Any) -> dict:
    """Write canned response to state as list[str]. No LLM call."""
    node_trace = list(state.get("node_trace", []))
    node_trace.append("out_of_scope")
    return {
        "answer": _OOS_RESPONSE,
        "answer_stream": [_OOS_RESPONSE],
        "node_trace": node_trace,
    }
