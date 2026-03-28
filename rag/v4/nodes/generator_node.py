"""Generator node — streams the final answer."""
from __future__ import annotations

from typing import Any

from rag.v4.middleware import error_guard_middleware, timing_middleware
from rag.v4.state import PipelineState


@timing_middleware
@error_guard_middleware
def generator_node(state: PipelineState, registry: Any) -> dict:
    """Generate the answer stream. Writes both answer_stream (Iterator) and answer (str)."""
    node_trace = list(state.get("node_trace", []))
    node_trace.append("generator")

    try:
        stream = registry.generator_llm.generate_stream(state)
        # Collect full answer for state (answer_stream is the live iterator for app.py)
        tokens = list(stream)
        answer = "".join(tokens)

        # Re-create iterator from collected tokens for app.py consumption
        def _replay():
            yield from tokens

        return {
            "answer": answer,
            "answer_stream": _replay(),
            "node_trace": node_trace,
        }
    except Exception as e:
        err_msg = f"Generation failed: {e}"
        return {
            "answer": err_msg,
            "answer_stream": iter([err_msg]),
            "error": err_msg,
            "node_trace": node_trace,
        }
