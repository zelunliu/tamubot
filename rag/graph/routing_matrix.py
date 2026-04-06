"""Data-driven routing matrix — reference documentation only.

This module is NOT wired to the LangGraph graph routing functions in builder.py.
The graph uses route_after_router() directly.
This matrix exists as a single source of truth for supported function types
and their retrieval strategies.
"""
from __future__ import annotations

from typing import TypedDict


class RoutingEntry(TypedDict):
    requires_retrieval: bool
    retrieval_passes: list[str]   # e.g. ["recursive_retrieval", "retrieval"] or ["hybrid_course"]
    generation_mode: str          # "stream" | "canned"


ROUTING_MATRIX: dict[str, RoutingEntry] = {
    "out_of_scope": {
        "requires_retrieval": False,
        "retrieval_passes": [],
        "generation_mode": "canned",
    },
    "recursive": {
        "requires_retrieval": True,
        "retrieval_passes": ["recursive_retrieval", "retrieval"],
        "generation_mode": "stream",
    },
    "hybrid_course": {
        "requires_retrieval": True,
        "retrieval_passes": ["hybrid_course"],
        "generation_mode": "stream",
    },
    "semantic_general": {
        "requires_retrieval": True,
        "retrieval_passes": ["semantic"],
        "generation_mode": "stream",
    },
}
