"""Generator node — streams the final answer."""
from __future__ import annotations

from typing import Any

from rag.v4.middleware import error_guard_middleware, timing_middleware
from rag.v4.state import PipelineState


@timing_middleware
@error_guard_middleware
def generator_node(state: PipelineState, registry: Any) -> dict:
    """Generate the answer. Stores answer_stream as list[str] (picklable for LangGraph)."""
    node_trace = list(state.get("node_trace", []))
    node_trace.append("generator")

    try:
        stream = registry.generator_llm.generate_stream(state)
        tokens = list(stream)
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
