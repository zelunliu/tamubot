"""Out-of-scope node — returns canned response without LLM call."""
from __future__ import annotations
from typing import Any
from rag.v4.state import PipelineState

_OOS_RESPONSE = (
    "Howdy! I'm TamuBot, your Texas A&M academic assistant. "
    "I can help you with questions about courses, syllabi, grading policies, "
    "schedules, and university policies. What would you like to know?"
)


def out_of_scope_node(state: PipelineState, registry: Any) -> dict:
    """Write canned response to state. No LLM call."""
    node_trace = list(state.get("node_trace", []))
    node_trace.append("out_of_scope")

    def _stream():
        yield _OOS_RESPONSE

    return {
        "answer": _OOS_RESPONSE,
        "answer_stream": _stream(),
        "node_trace": node_trace,
    }
