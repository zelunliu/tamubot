"""Recursive generator node — synthesizes new routing decision from anchor chunks.

Acts like a mini-router: given original query + history + anchor course chunks,
it decides what to search for next (function, course_ids, rewritten_query).
These fields overwrite state so retrieval_node runs with the resolved strategy.
"""
from __future__ import annotations

import json

from rag.graph.middleware import error_guard_middleware, timing_middleware
from rag.state.pipeline_state import PipelineState

_VALID_FUNCTIONS = {"semantic_general", "hybrid_course"}


@timing_middleware
@error_guard_middleware
def recursive_generator_node(state: PipelineState) -> dict:
    """Synthesize a new routing decision from anchor chunks + original query."""
    from rag.tools.llm import call_llm

    query = state.get("query", "")
    recursive_chunks = state.get("recursive_chunks", [])
    history_context = state.get("history_context") or ""
    course_ids = state.get("course_ids", [])
    node_trace = list(state.get("node_trace", []))
    node_trace.append("recursive_generator")

    # Summarize anchor content, capped to avoid token bloat
    anchor_text = " ".join(
        f"{c.get('header_text', '')} {c.get('content', '') or c.get('text', '')}"
        for c in recursive_chunks
    )[:2000]

    history_block = f"Conversation history:\n{history_context}\n\n" if history_context else ""
    course_label = ", ".join(course_ids) if course_ids else "the anchor course"

    prompt = (
        f"You are a query router for a Texas A&M University course search system.\n\n"
        f"The student originally asked: {query}\n\n"
        f"{history_block}"
        f"You retrieved the following information about {course_label}:\n{anchor_text}\n\n"
        f"Based on the student's intent and the retrieved course information, decide what to "
        f"search for next. Output JSON only:\n"
        f'{{"function": "semantic_general" or "hybrid_course", '
        f'"course_ids": [], '
        f'"rewritten_query": "targeted search string"}}\n\n'
        f'Use "hybrid_course" with specific course_ids only when the answer requires '
        f'looking up named specific courses (e.g. prerequisites). '
        f'Use "semantic_general" for discovery (similar topics, complementary courses, etc).'
    )

    try:
        llm_result = call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4096,
            json_mode=True,
        )
        data = json.loads(llm_result.text)

        new_function = data.get("function", "semantic_general")
        if new_function not in _VALID_FUNCTIONS:
            new_function = "semantic_general"

        raw_ids = data.get("course_ids") or []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        new_course_ids = [str(c).strip() for c in raw_ids if c]

        new_query = data.get("rewritten_query") or query

    except Exception:
        new_function = "semantic_general"
        new_course_ids = []
        new_query = query

    return {
        "function": new_function,
        "course_ids": new_course_ids,
        "rewritten_query": new_query,
        "node_trace": node_trace,
    }
