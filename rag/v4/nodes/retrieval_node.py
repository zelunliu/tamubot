"""Retrieval node — handles hybrid_course, semantic_general, and recurrent discover passes."""
from __future__ import annotations
from typing import Any
from rag.v4.state import PipelineState
from rag.router import compute_dynamic_k


def retrieval_node(state: PipelineState, registry: Any) -> dict:
    """Execute retrieval based on function type."""
    function = state.get("function", "out_of_scope")
    course_ids = state.get("course_ids", [])
    rewritten_query = state.get("rewritten_query") or state.get("query", "")
    eval_query = state.get("eval_query") or rewritten_query
    node_trace = list(state.get("node_trace", []))
    node_trace.append("retrieval")

    dk = compute_dynamic_k(function, len(course_ids))
    retrieve_k = dk["retrieve_k"]
    specific_categories = state.get("specific_categories", [])

    try:
        if function == "hybrid_course":
            all_chunks = []
            for cid in course_ids:
                chunks = registry.retriever.hybrid_search(rewritten_query, cid, retrieve_k)
                all_chunks.extend(chunks)
            reranked = registry.reranker.rerank(
                rewritten_query, all_chunks, top_k=len(all_chunks),
                specific_categories=specific_categories,
            )
            return {"retrieved_chunks": reranked, "node_trace": node_trace}

        elif function == "semantic_general":
            chunks = registry.retriever.semantic_search(rewritten_query, retrieve_k)
            reranked = registry.reranker.rerank(
                rewritten_query, chunks, top_k=len(chunks),
                specific_categories=specific_categories,
            )
            return {"retrieved_chunks": reranked, "node_trace": node_trace}

        elif function == "recurrent":
            # Discovery pass: semantic search excluding anchor courses
            anchor_ids = set(course_ids)
            all_results = registry.retriever.semantic_search(eval_query, retrieve_k)
            discovery_chunks = [c for c in all_results if c.get("course_id") not in anchor_ids]
            return {"discovery_chunks": discovery_chunks, "node_trace": node_trace}

        else:
            return {"retrieved_chunks": [], "node_trace": node_trace}

    except Exception as e:
        return {
            "retrieved_chunks": [],
            "discovery_chunks": [],
            "error": f"Retrieval failed: {e}",
            "node_trace": node_trace,
        }
