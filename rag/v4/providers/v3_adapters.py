"""v3 adapter implementations for the v4 ComponentRegistry protocols.

These thin wrappers call existing v3 functions so Phase 2 can run end-to-end
without touching the v3 codebase. Phase 3 replaces these with Haystack components.
"""
from __future__ import annotations

from typing import Any, Iterator, Optional


class V3RouterAdapter:
    """Wraps rag.router.classify_query() to satisfy RouterLLMComponent protocol."""

    def classify(
        self,
        query: str,
        trace: Optional[Any] = None,
        prior_course_ids: Optional[list[str]] = None,
    ) -> Any:
        from rag.router import classify_query
        # Pass trace as router_span for Langfuse observability
        return classify_query(query, router_span=trace, prior_course_ids=prior_course_ids)


class V3RetrieverAdapter:
    """Wraps rag.search_v3.* to satisfy RetrieverComponent protocol."""

    def hybrid_search(
        self, query: str, course_id: str, retrieve_k: int, embedding: Optional[list[float]] = None
    ) -> list[dict]:
        from rag.search_v3 import hybrid_search_v3
        return hybrid_search_v3(query, course_id=course_id, k=retrieve_k)

    def semantic_search(
        self, query: str, retrieve_k: int, embedding: Optional[list[float]] = None
    ) -> list[dict]:
        from rag.search_v3 import search_semantic
        return search_semantic(query, top_k=retrieve_k)

    def fetch_anchor_chunks(
        self, course_ids: list[str], categories: list[str]
    ) -> tuple[list[dict], list[tuple[str, str]], bool]:
        from rag.search_v3 import fetch_anchor_chunks
        return fetch_anchor_chunks(course_ids, categories)

    def get_meeting_times(self, course_ids: list[str]) -> dict[str, Any]:
        from rag.search_v3 import get_meeting_times
        return get_meeting_times(course_ids)

    def get_syllabus_urls(self, course_ids: list[str]) -> dict[str, str]:
        from rag.search_v3 import get_syllabus_urls
        return get_syllabus_urls(course_ids)


class V3RerankerAdapter:
    """Wraps rag.reranker.rerank() + stratified_select() to satisfy RerankerComponent protocol."""

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
        specific_categories: Optional[list[str]] = None,
    ) -> list[dict]:
        from rag import reranker
        reranked = reranker.rerank(query, chunks, top_k=top_k)
        return reranker.stratified_select(reranked, specific_categories or [])


class V3GeneratorAdapter:
    """Wraps rag.generator.* to satisfy GeneratorLLMComponent protocol."""

    def generate_stream(self, state: Any) -> Iterator[str]:
        from rag.generator import generate_stream
        router_result = state.get("router_result")
        return generate_stream(
            results=state.get("retrieved_chunks", []),
            question=state.get("rewritten_query") or state.get("query", ""),
            function=state.get("function", "semantic_general"),
            course_ids=state.get("course_ids", []),
            intent_type=state.get("intent_type"),
            specific_categories=state.get("specific_categories", []),
            specific_only=router_result.specific_only if router_result else False,
            data_gaps=state.get("data_gaps", []),
            data_integrity=state.get("data_integrity", True),
            conflicted_course_ids=state.get("conflicted_course_ids", []),
            trace=state.get("trace"),
        )

    def generate_eval_query(
        self, query: str, anchor_chunks: list[dict], trace: Optional[Any] = None
    ) -> str:
        from rag.generator import generate_eval_search_string
        # generate_eval_search_string signature: (anchor_chunks, original_query, intent_type, parent_span=None)
        return generate_eval_search_string(anchor_chunks, query, "GENERAL", parent_span=trace)
