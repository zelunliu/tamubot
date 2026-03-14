"""Tests for v4 Protocol contracts and ComponentRegistry."""
from rag.v4.interfaces import (
    RouterLLMComponent,
    RetrieverComponent,
    RerankerComponent,
    GeneratorLLMComponent,
    ComponentRegistry,
)


def test_all_protocols_importable():
    assert RouterLLMComponent is not None
    assert RetrieverComponent is not None
    assert RerankerComponent is not None
    assert GeneratorLLMComponent is not None


def test_component_registry_accepts_none():
    """ComponentRegistry(None, None, None, None) must not crash."""
    registry = ComponentRegistry(
        router_llm=None,
        retriever=None,
        reranker=None,
        generator_llm=None,
    )
    assert registry.router_llm is None
    assert registry.retriever is None
    assert registry.reranker is None
    assert registry.generator_llm is None


def test_component_registry_is_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(ComponentRegistry)
