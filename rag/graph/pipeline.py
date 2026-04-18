"""RAG pipeline entry point."""
from __future__ import annotations

import logging
from typing import Any, Optional

from rag.graph.builder import build_graph, build_graph_with_memory
from rag.state.pipeline_state import PipelineState

_logger = logging.getLogger("tamubot")

try:
    from langfuse.langchain import CallbackHandler
    from langfuse.types import TraceContext as _LFTraceContext
except ImportError:
    CallbackHandler = None  # type: ignore[assignment,misc]
    _LFTraceContext = None  # type: ignore[assignment,misc]

_graph = None
_eval_graph = None
_memory_graph = None

_INITIAL_STATE: dict = {
    "node_trace": [],
    "timing_ms": {},
    "data_gaps": [],
    "data_integrity": True,
    "recursive_chunks": [],
    "retrieved_chunks": [],
    "answer": "",
    "history_context": "",
    "rewritten_query": "",
    "answer_stream": None,
}


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _get_eval_graph():
    global _eval_graph
    if _eval_graph is None:
        from rag.graph.builder import build_graph_eval
        _eval_graph = build_graph_eval()
    return _eval_graph


def _make_invoke_kwargs(trace, thread_config: Optional[dict] = None) -> dict:
    """Build kwargs for graph.invoke(): callbacks from trace, config from thread_config."""
    callbacks = []
    if trace is not None and CallbackHandler is not None:
        try:
            callbacks = [CallbackHandler(trace_context=_LFTraceContext(trace_id=trace.trace_id))]
        except Exception:
            pass

    config: dict = {}
    if thread_config:
        config.update(thread_config)
    if callbacks:
        existing = config.get("callbacks", [])
        config["callbacks"] = existing + callbacks

    return {"config": config} if config else {}


def _build_router_result(result: dict):
    """Reconstruct a RouterResult from state fields for backward compat with app.py."""
    from rag.router import RouterResult
    return RouterResult(
        course_ids=result.get("course_ids", []),
        intent_type=result.get("intent_type"),
        recursive_search=result.get("recursive_search", False),
        rewritten_query=result.get("rewritten_query", ""),
        section=result.get("section"),
        function=result.get("function", "out_of_scope"),
        retrieval_mode=result.get("retrieval_mode", ""),
    )


def run_pipeline(
    query: str,
    trace=None,
    return_timing: bool = False,
) -> tuple:
    """Run the RAG pipeline (stateless).

    Returns:
        (chunks, router_result, data_gaps, data_integrity, conflicted_course_ids)
        or if return_timing=True: adds timing_ms dict as 6th element.
    """
    from rag.router import deduplicate_chunks

    initial_state: PipelineState = {**_INITIAL_STATE, "query": query}
    invoke_kwargs = _make_invoke_kwargs(trace)
    result = _get_graph().invoke(initial_state, **invoke_kwargs)

    anchor = result.get("recursive_chunks", [])
    followup = result.get("retrieved_chunks", [])
    chunks = deduplicate_chunks(anchor + followup) if anchor else followup

    five_tuple = (
        chunks,
        _build_router_result(result),
        result.get("data_gaps", []),
        result.get("data_integrity", True),
        [],  # conflicted_course_ids removed — schedule_filter no longer runs
    )
    if return_timing:
        return (*five_tuple, result.get("timing_ms", {}))
    return five_tuple


def run_pipeline_eval(
    query: str,
    trace=None,
) -> tuple[list[dict], Any, dict]:
    """Run router + retrieval only (no generator). For eval use.

    Same tracing as run_pipeline() — passes trace through CallbackHandler so
    every graph node (router, retrieval) appears as a child span in Langfuse.
    Session cache is disabled via SESSION_CACHE_ENABLED env var at eval time.

    Returns:
        (chunks, router_result, timing_ms)
    """
    from rag.router import deduplicate_chunks

    initial_state: PipelineState = {**_INITIAL_STATE, "query": query}
    invoke_kwargs = _make_invoke_kwargs(trace)
    result = _get_eval_graph().invoke(initial_state, **invoke_kwargs)

    anchor = result.get("recursive_chunks", [])
    followup = result.get("retrieved_chunks", [])
    combined = deduplicate_chunks(anchor + followup) if anchor else followup

    return (
        combined,
        _build_router_result(result),
        result.get("timing_ms", {}),
    )


def run_pipeline_with_memory(
    query: str,
    trace=None,
    thread_config: Optional[dict] = None,
) -> tuple[list[dict], Any, list[tuple[str, str]], bool, list[str], list[str]]:
    """Run the RAG pipeline with conversation memory.

    Returns 6-tuple: (chunks, router_result, data_gaps, data_integrity, conflicted_course_ids, answer_tokens)
    """
    global _memory_graph
    if _memory_graph is None:
        from rag.graph.checkpointer import make_checkpointer
        checkpointer = make_checkpointer()
        _memory_graph = build_graph_with_memory(checkpointer=checkpointer)

    # Explicitly load conversation state from checkpoint so history_inject_node
    # always sees prior turns regardless of LangGraph's checkpoint merge behavior.
    prior: dict = get_current_state(thread_config) if thread_config else {}
    _logger.info(
        "pipeline: pre-invoke history_len=%d summary_len=%d turn=%d",
        len(prior.get("history", [])),
        len(prior.get("history_summary", "") or ""),
        prior.get("turn_number", 0),
    )

    initial_state: dict = {
        **_INITIAL_STATE,
        "query": query,
        "history": prior.get("history", []),
        "history_summary": prior.get("history_summary", "") or "",
        "turn_number": prior.get("turn_number", 0),
    }

    if thread_config:
        session_id = thread_config.get("configurable", {}).get("thread_id", "")
        if session_id:
            initial_state["session_id"] = session_id

    invoke_kwargs = _make_invoke_kwargs(trace, thread_config)
    result = _memory_graph.invoke(initial_state, **invoke_kwargs)

    from rag.router import deduplicate_chunks

    anchor = result.get("recursive_chunks", [])
    followup = result.get("retrieved_chunks", [])
    chunks = deduplicate_chunks(anchor + followup) if anchor else followup

    answer_str = result.get("answer") or ""
    return (
        chunks,
        _build_router_result(result),
        result.get("data_gaps", []),
        result.get("data_integrity", True),
        [],  # conflicted_course_ids removed
        [answer_str] if answer_str else [],
    )


def get_current_state(thread_config: dict) -> dict:
    """Read conversation state for a thread. Returns empty dict if no state exists."""
    global _memory_graph
    if _memory_graph is None:
        return {}
    try:
        snapshot = _memory_graph.get_state(thread_config)
        return snapshot.values if snapshot and snapshot.values else {}
    except Exception:
        return {}
