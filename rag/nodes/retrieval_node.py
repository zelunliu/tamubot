"""Retrieval node — handles hybrid_course, semantic_general, and recurrent discover passes."""
from __future__ import annotations

import config
from rag.graph.cache_utils import normalize_query
from rag.graph.middleware import error_guard_middleware, timing_middleware, tracing_middleware
from rag.state.pipeline_state import PipelineState


def _compute_dynamic_k(function: str, n_courses: int) -> dict[str, int]:
    """Compute retrieve_k scaled by number of courses."""
    base = config.PER_COURSE_K[function]
    if function == "semantic_general":
        return dict(base)
    n = max(1, n_courses)
    return {
        "retrieve_k": min(base["retrieve_k"] * n, config.MAX_RETRIEVE_K),
        "rerank_k": min(base["rerank_k"] * n, config.MAX_RERANK_K),
    }


def _make_retrieval_cache_key(function, course_ids, rewritten_query, eval_query):
    if function == "recurrent":
        return f"recurrent|{normalize_query(eval_query)}"
    return f"{sorted(course_ids)}|{normalize_query(rewritten_query)}"


@tracing_middleware
@timing_middleware
@error_guard_middleware
def retrieval_node(state: PipelineState) -> dict:
    """Execute retrieval based on function type."""
    from rag.tools.mongo import hybrid_search, semantic_search
    from rag.tools.voyage import rerank as voyage_rerank

    function = state.get("function", "out_of_scope")
    course_ids = state.get("course_ids", [])
    rewritten_query = state.get("rewritten_query") or state.get("query", "")
    eval_query = state.get("eval_query") or rewritten_query
    node_trace = list(state.get("node_trace", []))
    node_trace.append("retrieval")

    dk = _compute_dynamic_k(function, len(course_ids))
    retrieve_k = dk["retrieve_k"]

    # Cache check — skip retrieval on exact-match hit
    if config.SESSION_CACHE_ENABLED:
        cache_key = _make_retrieval_cache_key(function, course_ids, rewritten_query, eval_query)
        cached_chunks = state.get("retrieval_cache", {}).get(cache_key)
        if cached_chunks is not None:
            node_trace.append("retrieval_cache_hit")
            if function == "recurrent":
                return {"discovery_chunks": cached_chunks, "node_trace": node_trace}
            return {"retrieved_chunks": cached_chunks, "node_trace": node_trace}

    try:
        if function == "hybrid_course":
            all_chunks = []
            for cid in course_ids:
                chunks = hybrid_search(rewritten_query, cid, retrieve_k)
                all_chunks.extend(chunks)
            reranked = voyage_rerank(
                rewritten_query, all_chunks, top_k=len(all_chunks),
            )

            retrieval_cache_update = {}
            if config.SESSION_CACHE_ENABLED:
                cache_key = _make_retrieval_cache_key(function, course_ids, rewritten_query, eval_query)
                existing_cache = state.get("retrieval_cache", {})
                retrieval_cache_update = {**existing_cache, cache_key: reranked}

            return {"retrieved_chunks": reranked, "retrieval_cache": retrieval_cache_update, "node_trace": node_trace}

        elif function == "semantic_general":
            chunks = semantic_search(rewritten_query, retrieve_k)
            reranked = voyage_rerank(
                rewritten_query, chunks, top_k=len(chunks),
            )

            retrieval_cache_update = {}
            if config.SESSION_CACHE_ENABLED:
                cache_key = _make_retrieval_cache_key(function, course_ids, rewritten_query, eval_query)
                existing_cache = state.get("retrieval_cache", {})
                retrieval_cache_update = {**existing_cache, cache_key: reranked}

            return {"retrieved_chunks": reranked, "retrieval_cache": retrieval_cache_update, "node_trace": node_trace}

        elif function == "recurrent":
            # Discovery pass: semantic search excluding anchor courses
            anchor_ids = set(course_ids)
            all_results = semantic_search(eval_query, retrieve_k)
            discovery_chunks = [c for c in all_results if c.get("course_id") not in anchor_ids]

            retrieval_cache_update = {}
            if config.SESSION_CACHE_ENABLED:
                cache_key = _make_retrieval_cache_key(function, course_ids, rewritten_query, eval_query)
                existing_cache = state.get("retrieval_cache", {})
                retrieval_cache_update = {**existing_cache, cache_key: discovery_chunks}

            return {
                "discovery_chunks": discovery_chunks,
                "retrieval_cache": retrieval_cache_update,
                "node_trace": node_trace,
            }

        else:
            return {"retrieved_chunks": [], "node_trace": node_trace}

    except Exception as e:
        return {
            "retrieved_chunks": [],
            "discovery_chunks": [],
            "error": f"Retrieval failed: {e}",
            "node_trace": node_trace,
        }
