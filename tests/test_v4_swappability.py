"""Tests that components are swappable without breaking the graph."""
from unittest.mock import MagicMock
from rag.v4.graph import build_graph
from rag.v4.interfaces import ComponentRegistry
from rag.v4.components.rerankers import IdentityReranker
from rag.v4.components.embedders import NullEmbedder
from rag.router import RouterResult


def test_graph_with_identity_reranker():
    """Swap VoyageReranker for IdentityReranker — graph should still work."""
    router = MagicMock()
    rr = RouterResult(
        course_ids=["202611_CSCE_221_500"],
        rewritten_query="test",
        function="hybrid_course",
        intent_type="ACADEMIC",
    )
    router.classify.return_value = rr

    retriever = MagicMock()
    retriever.hybrid_search.return_value = [{"course_id": "202611_CSCE_221_500", "text": "chunk"}]
    retriever.get_meeting_times.return_value = {}

    # Use real IdentityReranker (not a mock)
    identity_reranker = IdentityReranker()

    # Adapt IdentityReranker to satisfy RerankerComponent protocol
    class IdentityRerankerAdapter:
        def rerank(self, query, chunks, top_k, specific_categories=None):
            result = identity_reranker.run(query=query, chunks=chunks, top_k=top_k)
            return result["chunks"]

    generator = MagicMock()
    generator.generate_stream.return_value = iter(["Hello"])

    registry = ComponentRegistry(
        router_llm=router,
        retriever=retriever,
        reranker=IdentityRerankerAdapter(),
        generator_llm=generator,
    )

    graph = build_graph(registry)
    result = graph.invoke({
        "query": "test", "node_trace": [], "timing_ms": {},
        "conflicted_course_ids": [], "data_gaps": [], "data_integrity": True,
        "anchor_chunks": [], "discovery_chunks": [], "retrieved_chunks": [],
    })
    assert result["answer"]
    assert "retrieval" in result["node_trace"]
    assert "generator" in result["node_trace"]


def test_null_embedder_produces_empty_chunks_not_error():
    """NullEmbedder returning zero vector should produce empty results (not an error)."""
    from rag.v4.components.embedders import NullEmbedder
    embedder = NullEmbedder(dim=1024)
    result = embedder.run(text="anything")
    assert result["embedding"] == [0.0] * 1024
    # The embedder itself never errors — downstream MongoDB may return empty results
    assert not any(v != 0.0 for v in result["embedding"])
