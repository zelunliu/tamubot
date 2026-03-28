"""PipelineState TypedDict — the central data contract for the v4 LangGraph pipeline."""
from __future__ import annotations
from typing import Any, Iterator, Optional, TYPE_CHECKING
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from rag.router import RouterResult


class PipelineState(TypedDict, total=False):
    query: str
    router_result: Any          # RouterResult; typed as Any to avoid circular imports
    rewritten_query: str
    function: str               # "hybrid_course"|"recurrent"|"semantic_general"|"out_of_scope"
    course_ids: list[str]
    intent_type: Optional[str]
    specific_categories: list[str]
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
    answer_stream: Optional[Any]   # Iterator — NOT checkpointed
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
    turn_number: int
    router_cache: dict                           # normalize(query) → router_result_dict
    retrieval_cache: dict                        # cache_key → list[chunk_dict]
    answer_cache: dict                           # normalize(query) → answer str
