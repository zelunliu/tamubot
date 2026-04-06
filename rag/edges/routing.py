"""Conditional edge functions for the RAG graph."""
from __future__ import annotations

from rag.state.pipeline_state import PipelineState


def route_after_router(state: PipelineState) -> str:
    """Dispatch to retrieval path based on function type."""
    function = state.get("function", "out_of_scope")
    if function == "out_of_scope":
        return "out_of_scope"
    elif function == "recursive":
        return "recursive_retrieval"
    else:
        return "retrieval"
