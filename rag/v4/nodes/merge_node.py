"""Merge node — combines anchor + discovery chunks, deduplicates, reranks, caps courses."""
from __future__ import annotations
from typing import Any
import config
from rag.v4.state import PipelineState
from rag.router import deduplicate_chunks
from rag.v4.middleware import error_guard_middleware, timing_middleware


@timing_middleware
@error_guard_middleware
def merge_node(state: PipelineState, registry: Any) -> dict:
    """Merge anchor + discovery, rerank, cap at RECURRENT_MAX_RECOMMENDED_COURSES."""
    anchor_chunks = state.get("anchor_chunks", [])
    discovery_chunks = state.get("discovery_chunks", [])
    course_ids = state.get("course_ids", [])
    query = state.get("rewritten_query") or state.get("query", "")
    node_trace = list(state.get("node_trace", []))
    node_trace.append("merge")

    specific_categories = state.get("specific_categories", [])

    try:
        combined = deduplicate_chunks(anchor_chunks + discovery_chunks)
        reranked = registry.reranker.rerank(
            query, combined, top_k=len(combined),
            specific_categories=specific_categories,
        )

        # Cap discovery courses
        anchor_ids = set(course_ids)
        admitted: list[str] = []
        result = []
        for chunk in reranked:
            cid = chunk.get("course_id", "")
            if cid in anchor_ids:
                result.append(chunk)
            elif cid in admitted:
                result.append(chunk)
            elif len(admitted) < config.RECURRENT_MAX_RECOMMENDED_COURSES:
                admitted.append(cid)
                result.append(chunk)

        return {"retrieved_chunks": result, "node_trace": node_trace}
    except Exception as e:
        # Fall back to just anchor chunks
        return {
            "retrieved_chunks": anchor_chunks,
            "error": f"Merge failed: {e}",
            "node_trace": node_trace,
        }
