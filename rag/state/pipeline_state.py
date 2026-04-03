"""Pipeline state TypedDicts — data contracts for the RAG graph.

All data flowing between nodes is defined here. Nodes only read/write
these typed fields; no other shared state exists.

Canonical location: rag/state/pipeline_state.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from typing_extensions import TypedDict

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# RouterResult — structured output of the router node
# ---------------------------------------------------------------------------

def _derive_function(course_ids: list[str], recurrent_search: bool, intent_type: Optional[str]) -> str:
    if not course_ids:
        return "semantic_general" if intent_type is not None else "out_of_scope"
    return "recurrent" if recurrent_search else "hybrid_course"


def _derive_retrieval_mode(course_ids: list[str], recurrent_search: bool) -> str:
    if not course_ids:
        return "semantic"
    if recurrent_search:
        return "hybrid"
    return "hybrid_course"


@dataclass
class RouterResult:
    """Structured output from the query router."""
    course_ids: list[str] = field(default_factory=list)
    intent_type: Optional[str] = None
    recurrent_search: bool = False
    rewritten_query: str = ""
    section: Optional[str] = None
    retrieval_mode: str = ""
    function: str = ""

    def __post_init__(self):
        if not self.function:
            self.function = _derive_function(self.course_ids, self.recurrent_search, self.intent_type)
        if not self.retrieval_mode:
            self.retrieval_mode = _derive_retrieval_mode(self.course_ids, self.recurrent_search)

    @property
    def requires_retrieval(self) -> bool:
        return bool(self.course_ids) or self.intent_type is not None


def normalize_course_id(raw: str) -> str:
    """Normalize 'csce638' → 'CSCE 638'."""
    raw = raw.strip().upper().replace("-", " ")
    match = re.match(r"^([A-Z]+)\s*(\d+.*)$", raw)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return raw


# ---------------------------------------------------------------------------
# PipelineState — main state contract
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    query: str
    router_result: Any          # RouterResult; typed as Any to avoid circular imports
    rewritten_query: str
    function: str               # "hybrid_course"|"recurrent"|"semantic_general"|"out_of_scope"
    course_ids: list[str]
    intent_type: Optional[str]
    recurrent_search: bool
    requires_retrieval: bool
    anchor_chunks: list[dict]
    eval_query: str
    discovery_chunks: list[dict]
    retrieved_chunks: list[dict]
    data_gaps: list[tuple[str, str]]
    data_integrity: bool
    conflicted_course_ids: list[str]
    answer: str
    answer_stream: Optional[list]  # list[str] tokens — picklable, checkpointed by LangGraph
    trace: Optional[Any]           # LFTrace — NOT checkpointed (not picklable)
    timing_ms: dict[str, float]
    error: Optional[str]
    node_trace: list[str]


class ConversationMessage(TypedDict, total=False):
    role: str           # "user" | "assistant"
    content: str
    router_result: Optional[dict]   # serializable summary of RouterResult


class ConversationState(PipelineState, total=False):
    """Extends PipelineState with multi-turn session fields (Phase 5)."""
    session_id: str
    history: list[ConversationMessage]          # last N turns (windowed)
    history_summary: str                         # LLM-compressed older turns
    history_context: str          # formatted history block for generator (set by history_inject_node)
    turn_number: int
    router_cache: dict                           # normalize(query) → router_result_dict
    retrieval_cache: dict                        # cache_key → list[chunk_dict]
    answer_cache: dict                           # normalize(query) → answer str
