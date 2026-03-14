"""Factory functions for ComponentRegistry instances."""
from __future__ import annotations
from rag.v4.interfaces import ComponentRegistry


def make_v3_registry() -> ComponentRegistry:
    """Build a ComponentRegistry backed by v3 adapters (Phase 2 fallback)."""
    from rag.v4.providers.v3_adapters import (
        V3RouterAdapter,
        V3RetrieverAdapter,
        V3RerankerAdapter,
        V3GeneratorAdapter,
    )
    return ComponentRegistry(
        router_llm=V3RouterAdapter(),
        retriever=V3RetrieverAdapter(),
        reranker=V3RerankerAdapter(),
        generator_llm=V3GeneratorAdapter(),
    )


def make_default_registry() -> ComponentRegistry:
    """Build a ComponentRegistry backed by Haystack components (Phase 3 default).

    Uses MongoDocumentStore + VoyageEmbedder for retrieval,
    VoyageReranker for reranking, LLMRouterComponent for routing,
    LLMGeneratorComponent for generation.
    """
    from rag.v4.components.embedders import VoyageEmbedder
    from rag.v4.components.document_stores import MongoDocumentStore
    from rag.v4.components.retriever_adapter import MongoRetrieverAdapter
    from rag.v4.components.rerankers import VoyageReranker
    from rag.v4.components.routers import LLMRouterComponent
    from rag.v4.components.generators import LLMGeneratorComponent

    embedder = VoyageEmbedder()
    store = MongoDocumentStore()
    retriever = MongoRetrieverAdapter(store=store, embedder=embedder)
    reranker = VoyageReranker()
    router = LLMRouterComponent()
    generator = LLMGeneratorComponent()

    return ComponentRegistry(
        router_llm=router,
        retriever=retriever,
        reranker=reranker,
        generator_llm=generator,
    )
