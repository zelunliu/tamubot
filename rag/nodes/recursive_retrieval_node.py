"""Recursive retrieval node — first-pass hybrid search on anchor course(s)."""
from __future__ import annotations

import config
from rag.graph.cache_utils import normalize_query
from rag.graph.middleware import error_guard_middleware, timing_middleware
from rag.state.pipeline_state import PipelineState


def _compute_dynamic_k(n_courses: int) -> dict:
    """Compute retrieve_k and rerank_k for recursive path scaled by number of courses."""
    base_hybrid = config.PER_COURSE_K["hybrid_course"]
    base_recursive = config.PER_COURSE_K["recursive"]
    n = max(1, n_courses)
    return {
        "retrieve_k": min(base_hybrid["retrieve_k"] * n, config.MAX_RETRIEVE_K),
        "rerank_k": min(base_recursive["rerank_k"] * n, config.MAX_RERANK_K),
    }


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

    dk = _compute_dynamic_k(len(course_ids))
    retrieve_k = dk["retrieve_k"]
    rerank_k = dk["rerank_k"]

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
        reranked = voyage_rerank(rewritten_query, all_chunks, top_k=rerank_k)

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
