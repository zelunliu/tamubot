"""Recursive retrieval node — first-pass hybrid search on anchor course(s)."""
from __future__ import annotations

import config
from rag.graph.cache_utils import normalize_query
from rag.graph.middleware import error_guard_middleware, timing_middleware
from rag.state.pipeline_state import PipelineState


def _compute_dynamic_k(n_courses: int) -> int:
    """Compute retrieve_k for hybrid_course scaled by number of courses."""
    base = config.PER_COURSE_K["hybrid_course"]
    n = max(1, n_courses)
    return min(base["retrieve_k"] * n, config.MAX_RETRIEVE_K)


@timing_middleware
@error_guard_middleware
def recursive_retrieval_node(state: PipelineState) -> dict:
    """Fetch anchor course chunks via hybrid search. Only runs on recursive path."""
    from rag.tools.mongo import hybrid_search
    from rag.tools.voyage import rerank as voyage_rerank

    course_ids = state.get("course_ids", [])
    rewritten_query = state.get("rewritten_query") or state.get("query", "")
    node_trace = list(state.get("node_trace", []))
    node_trace.append("recursive_retrieval")

    retrieve_k = _compute_dynamic_k(len(course_ids))

    # Cache check
    if config.SESSION_CACHE_ENABLED:
        cache_key = f"recursive_anchor|{sorted(course_ids)}|{normalize_query(rewritten_query)}"
        cached = state.get("retrieval_cache", {}).get(cache_key)
        if cached is not None:
            node_trace.append("retrieval_cache_hit")
            return {"recursive_chunks": cached, "node_trace": node_trace}

    try:
        all_chunks = []
        for cid in course_ids:
            chunks = hybrid_search(rewritten_query, cid, retrieve_k)
            all_chunks.extend(chunks)
        reranked = voyage_rerank(rewritten_query, all_chunks, top_k=len(all_chunks))

        retrieval_cache_update = {}
        if config.SESSION_CACHE_ENABLED:
            cache_key = f"recursive_anchor|{sorted(course_ids)}|{normalize_query(rewritten_query)}"
            existing = state.get("retrieval_cache", {})
            retrieval_cache_update = {**existing, cache_key: reranked}

        return {
            "recursive_chunks": reranked,
            "retrieval_cache": retrieval_cache_update,
            "node_trace": node_trace,
        }
    except Exception as e:
        return {
            "recursive_chunks": [],
            "error": f"Recursive retrieval failed: {e}",
            "node_trace": node_trace,
        }
