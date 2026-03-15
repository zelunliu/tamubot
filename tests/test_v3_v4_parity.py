"""Parity test: same query → same function classification in v3 and v4 router.

Both the v3 path (classify_query directly), the v4 V3RouterAdapter, and the v4
LLMRouterComponent wrap the same underlying classify_query() call. These tests
verify that neither adapter layer introduces behavioural divergence in function
classification.

Note on test isolation: V3RouterAdapter uses lazy imports (`from rag.router import
classify_query` inside the method), so patching `rag.router.classify_query` is
sufficient. LLMRouterComponent uses a module-level binding — patching
`rag.v4.components.routers.classify_query` is required to intercept it, but due
to a known interaction with the Haystack @component decorator registry, that
patch can leak across tests in certain orderings. For that reason, the
LLMRouterComponent path is verified separately in test_v4_components.py
(test_llm_router_component_classify_with_stub) rather than here.
"""
import pytest
from unittest.mock import patch

from rag.router import RouterResult


# Sample queries covering all 4 function types.
# Each tuple: (query, expected_function, mock_router_result)
PARITY_QUERIES = [
    (
        "What is the grading breakdown for CSCE 221?",
        "hybrid_course",
        RouterResult(
            course_ids=["202611_CSCE_221_500"],
            rewritten_query="grading breakdown CSCE 221",
            function="hybrid_course",
            intent_type="ACADEMIC",
        ),
    ),
    (
        "what courses cover machine learning at TAMU?",
        "semantic_general",
        RouterResult(
            course_ids=[],
            rewritten_query="machine learning courses TAMU",
            function="semantic_general",
            intent_type="ACADEMIC",
        ),
    ),
    (
        "what's the weather today?",
        "out_of_scope",
        RouterResult(
            course_ids=[],
            rewritten_query="weather today",
            function="out_of_scope",
            intent_type=None,
        ),
    ),
    (
        "compare CSCE 221 with similar courses",
        "recurrent",
        RouterResult(
            course_ids=["202611_CSCE_221_500"],
            rewritten_query="compare CSCE 221 similar courses",
            function="recurrent",
            intent_type="ACADEMIC",
            recurrent_search=True,
        ),
    ),
]


@pytest.mark.parametrize("query,expected_function,mock_rr", PARITY_QUERIES)
def test_v3_v4_function_parity(query, expected_function, mock_rr):
    """v3 and v4 router must produce the same function classification for the same query.

    Both routers use the same underlying classify_query() — this test verifies
    the v4 adapter doesn't introduce any behavioral divergence vs the v3 path.

    Patches only rag.router.classify_query so the mock is confined to the
    rag.router module and restored cleanly without cross-test side effects.
    V3RouterAdapter and the v3 path both use lazy imports from rag.router,
    so a single patch location is sufficient and safe.
    """
    with patch("rag.router.classify_query", return_value=mock_rr) as mock_cq:
        # v3 path (lazy import from rag.router)
        from rag.router import classify_query as v3_classify
        v3_result = v3_classify(query)

        # v4 path via adapter (also lazy import from rag.router)
        from rag.v4.providers.v3_adapters import V3RouterAdapter
        adapter = V3RouterAdapter()
        v4_result = adapter.classify(query)

    assert v3_result.function == expected_function
    assert v4_result.function == expected_function


@pytest.mark.parametrize("query,expected_function,mock_rr", PARITY_QUERIES)
def test_v4_haystack_component_parity(query, expected_function, mock_rr):
    """LLMRouterComponent must produce the same function classification.

    Uses a separate patch target (rag.v4.components.routers.classify_query)
    since LLMRouterComponent holds a module-level binding. This test is kept
    separate from test_v3_v4_function_parity to avoid the Haystack component
    registry interaction that can cause cross-test pollution when both patches
    are stacked in a single parametrized test.
    """
    with patch("rag.v4.components.routers.classify_query", return_value=mock_rr):
        from rag.v4.components.routers import LLMRouterComponent
        component = LLMRouterComponent()
        v4_component_result = component.classify(query)

    assert v4_component_result.function == expected_function
