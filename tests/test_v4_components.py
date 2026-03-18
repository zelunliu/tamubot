"""Tests for Phase 3 Haystack components."""
from unittest.mock import MagicMock, patch


def test_null_embedder_returns_correct_shape():
    from rag.v4.components.embedders import NullEmbedder
    embedder = NullEmbedder(dim=1024)
    result = embedder.run(text="hello")
    assert "embedding" in result
    assert len(result["embedding"]) == 1024
    assert all(v == 0.0 for v in result["embedding"])


def test_identity_reranker_returns_chunks_unchanged():
    from rag.v4.components.rerankers import IdentityReranker
    reranker = IdentityReranker()
    chunks = [{"course_id": "A", "text": "a"}, {"course_id": "B", "text": "b"}]
    result = reranker.run(query="test", chunks=chunks, top_k=2)
    assert result["chunks"] == chunks


def test_identity_reranker_respects_top_k():
    from rag.v4.components.rerankers import IdentityReranker
    reranker = IdentityReranker()
    chunks = [{"text": str(i)} for i in range(5)]
    result = reranker.run(query="test", chunks=chunks, top_k=2)
    assert len(result["chunks"]) == 2


def test_mongo_document_store_instantiates_with_mock_client():
    from rag.v4.components.document_stores import MongoDocumentStore
    mock_client = MagicMock()
    store = MongoDocumentStore(mongo_client=mock_client)
    assert store._client is mock_client


def test_llm_router_component_classify_with_stub():
    from rag.v4.components.routers import LLMRouterComponent
    from rag.router import RouterResult

    stub_result = RouterResult(
        course_ids=[], rewritten_query="test", function="out_of_scope", intent_type=None
    )

    # LLMRouterComponent.classify ultimately calls classify_query — mock it
    with patch("rag.v4.components.routers.classify_query", return_value=stub_result):
        router = LLMRouterComponent()
        result = router.classify("what is the weather?")

    assert result.function == "out_of_scope"


def test_llm_router_component_uses_injected_llm_fn():
    """Injected llm_fn must be called instead of the default call_llm."""
    from rag.v4.components.routers import LLMRouterComponent

    stub_llm_fn = MagicMock()
    # Return a valid JSON response that classify_query can parse
    mock_response = MagicMock()
    mock_response.text = '{"course_ids": [], "rewritten_query": "stub query", "function": "out_of_scope", "intent_type": null, "specific_categories": [], "specific_only": false, "category_confidence": 0.0, "recurrent_search": false}'
    mock_response.input_tokens = None
    mock_response.output_tokens = None
    mock_response.thinking_tokens = None
    stub_llm_fn.return_value = mock_response

    router = LLMRouterComponent(llm_fn=stub_llm_fn)
    result = router.classify("what is the weather?")

    # Verify the stub was actually called (not bypassed)
    stub_llm_fn.assert_called_once()
    assert result.rewritten_query == "stub query"


def test_identity_reranker_rerank_method_satisfies_protocol():
    """IdentityReranker.rerank() must return list[dict] directly (not wrapped in dict)."""
    from rag.v4.components.rerankers import IdentityReranker
    reranker = IdentityReranker()
    chunks = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    result = reranker.rerank(query="test", chunks=chunks, top_k=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"text": "a"}


def test_all_components_importable():
    from rag.v4.components.embedders import VoyageEmbedder, NullEmbedder
    from rag.v4.components.rerankers import VoyageReranker, IdentityReranker
    from rag.v4.components.document_stores import MongoDocumentStore
    from rag.v4.components.routers import LLMRouterComponent
    from rag.v4.components.generators import LLMGeneratorComponent
    from rag.v4.components.retriever_adapter import MongoRetrieverAdapter
    assert True  # no ImportError


def test_llm_generator_component_satisfy_protocol():
    from rag.v4.components.generators import LLMGeneratorComponent

    mock_stream_fn = MagicMock(return_value=iter(["Hello"]))
    gen = LLMGeneratorComponent(stream_fn=mock_stream_fn)

    state = {
        "query": "test",
        "rewritten_query": "test",
        "function": "hybrid_course",
        "course_ids": ["CSCE_221"],
        "retrieved_chunks": [],
        "data_gaps": [],
        "data_integrity": True,
        "conflicted_course_ids": [],
        "specific_categories": [],
        "intent_type": None,
        "router_result": None,
        "trace": None,
    }
    stream = gen.generate_stream(state)
    tokens = list(stream)
    assert tokens == ["Hello"]


def test_llm_router_restores_call_llm_after_run():
    """LLMRouterComponent.run() must restore rag.router.call_llm after injection."""
    import rag.router as _router_mod
    from rag.v4.components.routers import LLMRouterComponent
    from rag.router import RouterResult

    original_call_llm = _router_mod.call_llm

    # Create a mock LLM function that returns a proper response object
    fake_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"course_ids": [], "rewritten_query": "test", "function": "out_of_scope", "intent_type": null, "specific_categories": [], "specific_only": false, "category_confidence": 0.0, "recurrent_search": false}'
    mock_response.input_tokens = None
    mock_response.output_tokens = None
    fake_llm.return_value = mock_response

    component = LLMRouterComponent(llm_fn=fake_llm)
    component.run(query="test")

    assert _router_mod.call_llm is original_call_llm, \
        "call_llm must be restored to original after LLMRouterComponent.run()"
