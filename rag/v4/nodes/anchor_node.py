"""Anchor node — fetches all chunks for anchor course(s) in the recurrent pass."""
from __future__ import annotations

from typing import Any

from rag.v4.middleware import error_guard_middleware, timing_middleware
from rag.v4.state import PipelineState


@timing_middleware
@error_guard_middleware
def anchor_node(state: PipelineState, registry: Any) -> dict:
    """Fetch anchor chunks. Only runs for recurrent path."""
    course_ids = state.get("course_ids", [])
    node_trace = list(state.get("node_trace", []))
    node_trace.append("anchor")

    try:
        specific_categories = state.get("specific_categories", [])
        chunks, data_gaps, data_integrity = registry.retriever.fetch_anchor_chunks(course_ids, specific_categories)
        return {
            "anchor_chunks": chunks,
            "data_gaps": data_gaps,
            "data_integrity": data_integrity,
            "node_trace": node_trace,
        }
    except Exception as e:
        return {
            "anchor_chunks": [],
            "data_gaps": [],
            "data_integrity": False,
            "error": f"Anchor fetch failed: {e}",
            "node_trace": node_trace,
        }
