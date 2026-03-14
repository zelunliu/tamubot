"""Data-driven routing matrix — replaces if/else function dispatch.

Each entry defines what retrieval passes are needed and how to generate.
Keys match the function values derived by router._derive_function().
"""
from __future__ import annotations
from typing import TypedDict


class RoutingEntry(TypedDict):
    requires_retrieval: bool
    retrieval_passes: list[str]   # e.g. ["anchor", "discover"] or ["hybrid_course"]
    generation_mode: str          # "stream" | "canned"


ROUTING_MATRIX: dict[str, RoutingEntry] = {
    "out_of_scope": {
        "requires_retrieval": False,
        "retrieval_passes": [],
        "generation_mode": "canned",
    },
    "recurrent": {
        "requires_retrieval": True,
        "retrieval_passes": ["anchor", "discover"],
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
