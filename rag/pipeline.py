"""Explicit orchestration pipeline for TamuBot's 3-stage RAG flow.

Defines three named functions that make the retrieval loop visible at the
orchestration level rather than buried inside router.py:

  router_order    — Router_Stage: classify query → RouterResult
  db_order        — Retrieval_Stage: anchor or discover pass → chunks
  generator_order — Generator_Stage: eval search string or final answer stream

run_pipeline() composes these into the full pipeline, preserving the Langfuse
span hierarchy from the original router.py implementation.
"""
from __future__ import annotations

from typing import Iterator, Optional

import config
from rag import reranker
from rag import search_v3 as search
from rag.router import (
    FUNCTION_CATEGORY_STRATEGIES,
    RouterResult,
    classify_query,
    compute_dynamic_k,
    deduplicate_chunks,
)
from rag.search_v3 import fetch_anchor_chunks


def router_order(query: str, trace=None) -> RouterResult:
    """Router_Stage: open span, classify query, close span, return RouterResult."""
    router_span = None
    if trace is not None:
        try:
            router_span = trace.span(
                name="Router_Stage",
                input={"query": query},
            )
        except Exception:
            router_span = None

    router_result = classify_query(query, router_span=router_span)

    if router_span is not None:
        try:
            router_span.end()
        except Exception:
            pass

    return router_result


def db_order(
    pass_type: str,
    router_result: RouterResult,
    discovery_query: Optional[str] = None,
    trace=None,
) -> tuple[list[dict], list[tuple[str, str]], bool]:
    """Retrieval pass: 'anchor' (metadata fetch) or 'discover' (vector search).

    Args:
        pass_type:        "anchor" or "discover"
        router_result:    RouterResult from router_order()
        discovery_query:  Override search string for the discover pass (recurrent eval query)
        trace:            Parent span (Retrieval_Stage) forwarded to search/reranker calls

    Returns:
        (chunks, data_gaps, data_integrity)
        data_gaps and data_integrity are populated only for the anchor pass.
    """
    fn = router_result.function
    dk = compute_dynamic_k(fn, len(router_result.course_ids))
    retrieve_k = dk["retrieve_k"]
    rerank_k = dk["rerank_k"]

    if pass_type == "anchor":
        strategy = FUNCTION_CATEGORY_STRATEGIES.get(fn)
        effective_cats = strategy(router_result) if strategy else list(config.DEFAULT_SUMMARY_CATEGORIES)
        chunks, data_gaps, integrity = fetch_anchor_chunks(router_result.course_ids, effective_cats)
        return chunks, data_gaps, integrity

    # discover pass
    search_q = discovery_query or router_result.rewritten_query

    if fn == "semantic_general":
        results = search.search_semantic(search_q, top_k=retrieve_k)
        reranked = reranker.rerank(search_q, results, top_k=rerank_k, parent_span=trace)
        return reranked, [], True

    if fn == "out_of_scope" or not router_result.course_ids:
        return [], [], True

    # recurrent discover: semantic search with rewritten query, filter anchor course_ids, rerank
    all_results = search.search_semantic(search_q, top_k=retrieve_k)
    anchor_ids = set(router_result.course_ids)
    discovery_chunks = [c for c in all_results if c.get("course_id") not in anchor_ids]
    reranked = reranker.rerank(search_q, discovery_chunks, top_k=rerank_k, parent_span=trace)
    return reranked, [], True


def generator_order(
    recurrent: bool,
    chunks: list[dict],
    query: str,
    router_result: RouterResult,
    data_gaps: Optional[list] = None,
    data_integrity: bool = True,
    trace=None,
) -> "str | Iterator[str]":
    """Generator_Stage: eval search string (recurrent) or answer stream (final).

    Args:
        recurrent:      True → return eval search string (str); False → return answer stream
        chunks:         Retrieved + reranked chunks
        query:          User query (or rewritten_query) passed to the generator
        router_result:  RouterResult with function, course_ids, intent_type, etc.
        data_gaps:      From anchor pass
        data_integrity: From anchor pass
        trace:          Langfuse trace for Generator_Stage span (non-recurrent only)

    Returns:
        str (recurrent=True) or Iterator[str] (recurrent=False)
    """
    if recurrent:
        from rag.generator import generate_eval_search_string
        return generate_eval_search_string(
            chunks,
            router_result.rewritten_query or query,
            router_result.intent_type or "GENERAL",
        )

    from rag.generator import generate_stream
    return generate_stream(
        results=chunks,
        question=query,
        function=router_result.function,
        course_ids=router_result.course_ids,
        intent_type=router_result.intent_type,
        data_gaps=data_gaps or [],
        data_integrity=data_integrity,
        trace=trace,
    )


def run_pipeline(
    query: str,
    trace=None,
) -> tuple[list[dict], RouterResult, list[tuple[str, str]], bool]:
    """Full 3-stage pipeline: router_order → db_order(s) → ready for generator_order.

    Langfuse span hierarchy preserved:
      TamuBot_Complete_Pipeline (trace)
        └─ Router_Stage         ← router_order()
        └─ Retrieval_Stage      ← run_pipeline() (wraps all db_order calls)
             └─ Voyage_Embeddings, MongoDB_*, Voyage_Reranker

    Args:
        query: Raw user question.
        trace: Optional Langfuse trace.

    Returns:
        (reranked_chunks, router_result, data_gaps, data_integrity)
    """
    router_result = router_order(query, trace)
    search_query = router_result.rewritten_query or query

    if not router_result.requires_retrieval:
        return [], router_result, [], True

    # --- Retrieval_Stage span ---
    retrieval_span = None
    if trace is not None:
        try:
            retrieval_span = trace.span(
                name="Retrieval_Stage",
                input={
                    "query": search_query,
                    "function": router_result.function,
                    "retrieval_mode": router_result.retrieval_mode,
                    "course_ids": router_result.course_ids,
                },
            )
        except Exception:
            retrieval_span = None

    data_gaps: list[tuple[str, str]] = []
    data_integrity = True

    try:
        if not router_result.course_ids:
            # semantic_general: corpus-wide vector search
            chunks, _, _ = db_order("discover", router_result, trace=retrieval_span)
            reranked = deduplicate_chunks(chunks)
        else:
            anchor, data_gaps, data_integrity = db_order("anchor", router_result, trace=retrieval_span)
            if router_result.recurrent_search:
                # Skip eval_q generation — use rewritten query directly for
                # corpus-wide semantic discovery (avoids anchoring to anchor course)
                discovery, _, _ = db_order(
                    "discover", router_result, discovery_query=None, trace=retrieval_span
                )
                reranked = deduplicate_chunks(anchor + discovery)
            else:
                reranked = deduplicate_chunks(anchor)
    except Exception as e:
        if retrieval_span is not None:
            try:
                retrieval_span.end(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise

    if retrieval_span is not None:
        try:
            retrieval_span.end(output={
                "n_results": len(reranked),
                "data_gaps": len(data_gaps),
                "data_integrity": data_integrity,
            })
        except Exception:
            pass

    return reranked, router_result, data_gaps, data_integrity
