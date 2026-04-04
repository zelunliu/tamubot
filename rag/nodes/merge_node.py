"""Merge node — combines anchor + discovery chunks, deduplicates, reranks, caps courses."""
from __future__ import annotations

import config
from rag.graph.middleware import error_guard_middleware, timing_middleware
from rag.state.pipeline_state import PipelineState


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Remove duplicate chunks by (course_id, chunk_index)."""
    seen: set[tuple] = set()
    result = []
    for c in chunks:
        key = (c.get("course_id", ""), c.get("chunk_index", ""))
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result


@timing_middleware
@error_guard_middleware
def merge_node(state: PipelineState) -> dict:
    """Merge anchor + discovery, rerank, cap at RECURRENT_MAX_RECOMMENDED_COURSES."""
    from rag.tools.voyage import rerank

    anchor_chunks = state.get("anchor_chunks", [])
    discovery_chunks = state.get("discovery_chunks", [])
    course_ids = state.get("course_ids", [])
    query = state.get("rewritten_query") or state.get("query", "")
    node_trace = list(state.get("node_trace", []))
    node_trace.append("merge")

    try:
        combined = _deduplicate_chunks(anchor_chunks + discovery_chunks)
        reranked = rerank(query, combined, top_k=len(combined))

        # Cap discovery courses
        anchor_ids = set(course_ids)
        admitted: set[str] = set()
        result = []
        for chunk in reranked:
            cid = chunk.get("course_id", "")
            if cid in anchor_ids:
                result.append(chunk)
            elif cid in admitted:
                result.append(chunk)
            elif len(admitted) < config.RECURRENT_MAX_RECOMMENDED_COURSES:
                admitted.add(cid)
                result.append(chunk)

        return {"retrieved_chunks": result, "node_trace": node_trace}
    except Exception as e:
        # Fall back to just anchor chunks
        return {
            "retrieved_chunks": anchor_chunks,
            "error": f"Merge failed: {e}",
            "node_trace": node_trace,
        }
