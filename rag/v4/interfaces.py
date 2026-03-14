"""Component protocols and ComponentRegistry for the v4 pipeline.

These protocols define the method contracts each provider must satisfy.
Swap any field in ComponentRegistry → different provider, zero graph changes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Protocol, runtime_checkable


@runtime_checkable
class RouterLLMComponent(Protocol):
    def classify(self, query: str, trace: Optional[Any] = None) -> Any:
        """Classify a query and return a RouterResult."""
        ...


@runtime_checkable
class RetrieverComponent(Protocol):
    def hybrid_search(
        self, query: str, course_id: str, retrieve_k: int, embedding: Optional[list[float]] = None
    ) -> list[dict]:
        """Hybrid (vector + BM25) search filtered to one course."""
        ...

    def semantic_search(self, query: str, retrieve_k: int, embedding: Optional[list[float]] = None) -> list[dict]:
        """Corpus-wide semantic search."""
        ...

    def fetch_anchor_chunks(
        self, course_ids: list[str], categories: list[str]
    ) -> tuple[list[dict], list[tuple[str, str]], bool]:
        """Fetch anchor chunks for recurrent search. Returns (chunks, data_gaps, data_integrity)."""
        ...

    def get_meeting_times(self, course_ids: list[str]) -> dict[str, Any]:
        """Return meeting time data for schedule conflict filtering."""
        ...

    def get_syllabus_urls(self, course_ids: list[str]) -> dict[str, str]:
        """Return syllabus PDF URLs keyed by course_id."""
        ...


@runtime_checkable
class RerankerComponent(Protocol):
    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        """Rerank chunks by relevance to query, return top_k."""
        ...


@runtime_checkable
class GeneratorLLMComponent(Protocol):
    def generate_stream(self, state: Any) -> Iterator[str]:
        """Stream answer tokens given current pipeline state."""
        ...

    def generate_eval_query(self, query: str, anchor_chunks: list[dict], trace: Optional[Any] = None) -> str:
        """Generate the eval/discovery search string for the recurrent pass."""
        ...


@dataclass
class ComponentRegistry:
    """Holds one instance of each provider. Injected into every graph node.

    Swap any field → different provider, zero graph changes.
    """
    router_llm: Optional[RouterLLMComponent]
    retriever: Optional[RetrieverComponent]
    reranker: Optional[RerankerComponent]
    generator_llm: Optional[GeneratorLLMComponent]
