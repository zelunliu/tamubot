"""Factory functions for ComponentRegistry instances."""
from __future__ import annotations
from rag.v4.interfaces import ComponentRegistry


def make_v3_registry() -> ComponentRegistry:
    """Build a ComponentRegistry backed by v3 adapters.

    This is the Phase 2 default. Phase 3 adds make_default_registry() using
    Haystack components.
    """
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
