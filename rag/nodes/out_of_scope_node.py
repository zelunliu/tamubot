"""Out-of-scope node — returns canned response without LLM call."""
from __future__ import annotations

from rag.graph.middleware import error_guard_middleware, timing_middleware
from rag.state.pipeline_state import PipelineState

_OOS_RESPONSE = (
    "Howdy! I'm TamuBot, your Texas A&M academic assistant. "
    "I can help you with questions about courses, syllabi, grading policies, "
    "schedules, and university policies. What would you like to know?"
)


@timing_middleware
@error_guard_middleware
def out_of_scope_node(state: PipelineState) -> dict:
    """Write canned response to state as list[str]. No LLM call."""
    node_trace = list(state.get("node_trace", []))
    node_trace.append("out_of_scope")
    return {
        "answer": _OOS_RESPONSE,
        "answer_stream": [_OOS_RESPONSE],
        "node_trace": node_trace,
    }
