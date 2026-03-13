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
    RouterResult,
    classify_query,
    compute_dynamic_k,
    deduplicate_chunks,
)
from rag.search_v3 import fetch_anchor_chunks


def router_order(query: str, trace=None) -> RouterResult:
    """Router_Stage: open generation, classify query (generation closed inside classify_query)."""
    from rag.prompts import ROUTER_PROMPT

    router_obs = None
    if trace is not None:
        try:
            router_obs = trace.generation(
                name="Router_Stage",
                model=config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL,
                input=[{"role": "user", "content": ROUTER_PROMPT.format(query=query)}],
            )
        except Exception:
            router_obs = None

    # classify_query calls router_obs.end() with usage counts
    return classify_query(query, router_span=router_obs)


def db_order(
    pass_type: str,
    router_result: RouterResult,
    discovery_query: Optional[str] = None,
    trace=None,
) -> tuple[list[dict], list[tuple[str, str]], bool]:
    """Retrieval pass: 'anchor', 'hybrid_course', or 'discover'.

    Args:
        pass_type:        "anchor" | "hybrid_course" | "discover"
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
    search_q = discovery_query or router_result.rewritten_query

    if pass_type == "anchor":
        # Recurrent anchor: fetch all chunks for anchor course(s) — categories ignored in v3
        chunks, data_gaps, integrity = fetch_anchor_chunks(router_result.course_ids, [])
        return chunks, data_gaps, integrity

    if pass_type == "hybrid_course":
        # Per-course filtered hybrid search (vector + BM25), then cross-course rerank
        all_chunks: list[dict] = []
        for cid in router_result.course_ids:
            course_chunks = search.hybrid_search_v3(
                search_q, course_id=cid, k=retrieve_k, parent_span=trace
            )
            all_chunks.extend(course_chunks)
        all_reranked = reranker.rerank(search_q, all_chunks, top_k=len(all_chunks), parent_span=trace)
        reranked = reranker.stratified_select(all_reranked, router_result.specific_categories)
        return reranked, [], True

    # discover pass (recurrent discovery or semantic_general)
    if fn == "semantic_general":
        results = search.search_semantic(search_q, top_k=retrieve_k, parent_span=trace)
        all_reranked = reranker.rerank(search_q, results, top_k=len(results), parent_span=trace)
        reranked = reranker.stratified_select(all_reranked, router_result.specific_categories)
        return reranked, [], True

    if fn == "out_of_scope" or not router_result.course_ids:
        return [], [], True

    # recurrent discover: semantic search with rewritten query, filter anchor course_ids, rerank
    all_results = search.search_semantic(search_q, top_k=retrieve_k, parent_span=trace)
    anchor_ids = set(router_result.course_ids)
    discovery_chunks = [c for c in all_results if c.get("course_id") not in anchor_ids]
    all_reranked = reranker.rerank(search_q, discovery_chunks, top_k=len(discovery_chunks), parent_span=trace)
    reranked = reranker.stratified_select(all_reranked, router_result.specific_categories)
    return reranked, [], True


def _apply_schedule_filter(
    anchor_course_ids: list[str],
    discovery_chunks: list[dict],
) -> tuple[list[dict], list[str]]:
    """Fetch meeting times from courses_v3 and remove conflicting discovery courses."""
    from rag.schedule import filter_conflicting_courses, parse_meeting_times
    from rag.search_v3 import get_meeting_times

    anchor_mt_map = get_meeting_times(anchor_course_ids)
    anchor_interval = None
    for cid in anchor_course_ids:
        anchor_interval = parse_meeting_times(anchor_mt_map.get(cid))
        if anchor_interval:
            break

    if anchor_interval is None:
        return discovery_chunks, []  # anchor is async — nothing to filter

    disc_cids = list({c.get("course_id") for c in discovery_chunks if c.get("course_id")})
    disc_mt_map = get_meeting_times(disc_cids)
    return filter_conflicting_courses(discovery_chunks, anchor_interval, disc_mt_map)


def _cap_discovery_courses(
    chunks: list[dict],
    anchor_course_ids: set[str],
    max_courses: int,
) -> list[dict]:
    """Keep all anchor chunks + chunks from at most max_courses unique discovery courses."""
    admitted: list[str] = []
    result = []
    for chunk in chunks:
        cid = chunk.get("course_id", "")
        if cid in anchor_course_ids:
            result.append(chunk)
        elif cid in admitted:
            result.append(chunk)
        elif len(admitted) < max_courses:
            admitted.append(cid)
            result.append(chunk)
    return result


def generator_order(
    recurrent: bool,
    chunks: list[dict],
    query: str,
    router_result: RouterResult,
    data_gaps: Optional[list] = None,
    data_integrity: bool = True,
    conflicted_course_ids: Optional[list] = None,
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
        specific_categories=router_result.specific_categories,
        specific_only=router_result.specific_only,
        data_gaps=data_gaps or [],
        data_integrity=data_integrity,
        conflicted_course_ids=conflicted_course_ids or [],
        trace=trace,
    )


def run_pipeline(
    query: str,
    trace=None,
) -> tuple[list[dict], RouterResult, list[tuple[str, str]], bool, list[str]]:
    """Full pipeline: router_order → db_order(s) → ready for generator_order.

    Langfuse trace hierarchy:
      Non-recurrent / semantic_general:
        TamuBot_Complete_Pipeline
          ├─ Router_Stage
          ├─ Retrieval_Stage  (+ Voyage_Embeddings, MongoDB_*, Voyage_Reranker inside)
          └─ Generator_Stage

      Recurrent (6 flat steps):
        TamuBot_Complete_Pipeline
          ├─ Router_Stage
          ├─ Anchor_Pass       (direct MongoDB fetch, no embeddings)
          ├─ EvalSearch_Stage  (LLM generates discovery query from anchor content)
          ├─ Discover_Pass     (+ Voyage_Embeddings, MongoDB_*, Voyage_Reranker inside)
          ├─ Schedule_Filter   (removes courses with conflicting meeting times)
          └─ Generator_Stage

    Args:
        query: Raw user question.
        trace: Optional Langfuse trace.

    Returns:
        (reranked_chunks, router_result, data_gaps, data_integrity, conflicted_course_ids)
    """
    router_result = router_order(query, trace)
    search_query = router_result.rewritten_query or query

    if not router_result.requires_retrieval:
        return [], router_result, [], True, []

    dk = compute_dynamic_k(router_result.function, len(router_result.course_ids))
    data_gaps: list[tuple[str, str]] = []
    data_integrity = True

    def _span(name, input=None):
        if trace is None:
            return None
        try:
            return trace.span(name=name, input=input)
        except Exception:
            return None

    def _end_span(span, **kwargs):
        if span is None:
            return
        try:
            span.end(**kwargs)
        except Exception:
            pass

    def _results_summary(chunks):
        return [
            {"course_id": c.get("course_id"), "chunk_index": c.get("chunk_index"), "score": c.get("score")}
            for c in chunks
        ]

    conflicted_ids: list[str] = []

    try:
        if router_result.recurrent_search:
            # ── Recurrent: 6 flat steps directly under the trace ──────────────
            # Step 2: Anchor_Pass — direct MongoDB lookup, no embeddings
            anchor_span = _span("Anchor_Pass", input={"course_ids": router_result.course_ids})
            anchor, data_gaps, data_integrity = db_order("anchor", router_result)
            _end_span(anchor_span, output={"n_chunks": len(anchor), "data_gaps": len(data_gaps)})

            # Step 3: EvalSearch_Stage — LLM call to build discovery query
            from rag.generator import generate_eval_search_string
            eval_query = generate_eval_search_string(
                anchor,
                router_result.rewritten_query or search_query,
                router_result.intent_type or "GENERAL",
                parent_span=trace,
            )

            # Step 4: Discover_Pass — semantic search + rerank with eval query
            discover_span = _span("Discover_Pass", input={"query": eval_query})
            discovery, _, _ = db_order(
                "discover", router_result, discovery_query=eval_query,
                trace=discover_span or trace,
            )
            _end_span(discover_span, output={
                "n_chunks": len(discovery), "results_summary": _results_summary(discovery),
            })

            # Step 5: Schedule_Filter — remove courses conflicting with anchor schedule
            filter_span = _span("Schedule_Filter", input={"anchor_course_ids": router_result.course_ids})
            discovery, conflicted_ids = _apply_schedule_filter(router_result.course_ids, discovery)
            _end_span(filter_span, output={
                "n_after_filter": len(discovery),
                "conflicted_course_ids": conflicted_ids,
            })

            combined = deduplicate_chunks(anchor + discovery)
            all_reranked = reranker.rerank(search_query, combined, top_k=len(combined), parent_span=trace)
            reranked = reranker.stratified_select(all_reranked, router_result.specific_categories)
            reranked = _cap_discovery_courses(
                reranked,
                anchor_course_ids=set(router_result.course_ids),
                max_courses=config.RECURRENT_MAX_RECOMMENDED_COURSES,
            )

        elif not router_result.course_ids:
            # ── semantic_general: corpus-wide vector search ────────────────────
            retrieval_span = _span("Retrieval_Stage", input={
                "query": search_query, "function": router_result.function,
                "retrieve_k": dk["retrieve_k"], "rerank_k": dk["rerank_k"],
            })
            chunks, _, _ = db_order("discover", router_result, trace=retrieval_span)
            reranked = deduplicate_chunks(chunks)
            _end_span(retrieval_span, output={
                "n_results": len(reranked), "results_summary": _results_summary(reranked),
            })

        else:
            # ── hybrid_course: per-course filtered hybrid search + cross-course rerank ──
            retrieval_span = _span("Retrieval_Stage", input={
                "query": search_query, "function": router_result.function,
                "retrieval_mode": router_result.retrieval_mode,
                "course_ids": router_result.course_ids,
                "retrieve_k": dk["retrieve_k"], "rerank_k": dk["rerank_k"],
            })
            chunks, data_gaps, data_integrity = db_order(
                "hybrid_course", router_result, trace=retrieval_span
            )
            reranked = deduplicate_chunks(chunks)
            _end_span(retrieval_span, output={
                "n_results": len(reranked), "results_summary": _results_summary(reranked),
            })

    except Exception:
        raise

    return reranked, router_result, data_gaps, data_integrity, conflicted_ids
