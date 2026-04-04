"""Conditional edge functions for the RAG graph.

Each function reads one state field and returns a node name string.

Canonical location: rag/edges/routing.py
"""
from __future__ import annotations

from rag.state.pipeline_state import PipelineState


def route_after_router(state: PipelineState) -> str:
    """Dispatch to retrieval path based on function type."""
    function = state.get("function", "out_of_scope")
    if function == "out_of_scope":
        return "out_of_scope"
    elif function == "recurrent":
        return "anchor"
    else:
        return "retrieval"


def route_after_retrieval(state: PipelineState) -> str:
    """After retrieval: recurrent → schedule_filter, others → generator."""
    function = state.get("function", "out_of_scope")
    if function == "recurrent":
        return "schedule_filter"
    return "generator"
