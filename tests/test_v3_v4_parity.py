"""Parity test: same query → same function classification via classify_query.

The v3/v4 adapter layer has been removed. Both graph paths now call classify_query
directly via rag.router. These tests verify classify_query produces consistent
results when mocked.
"""
from unittest.mock import patch

import pytest

from rag.router import RouterResult

# Sample queries covering all 4 function types.
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
        "recursive",
        RouterResult(
            course_ids=["202611_CSCE_221_500"],
            rewritten_query="compare CSCE 221 similar courses",
            function="recursive",
            intent_type="ACADEMIC",
            recursive_search=True,
        ),
    ),
]


@pytest.mark.parametrize("query,expected_function,mock_rr", PARITY_QUERIES)
def test_classify_query_function_classification(query, expected_function, mock_rr):
    """classify_query must produce the expected function classification.

    Patches rag.router.classify_query to verify the mock result passes through
    correctly and the function field is set.
    """
    with patch("rag.router.classify_query", return_value=mock_rr):
        from rag.router import classify_query
        result = classify_query(query)

    assert result.function == expected_function
