"""Tests that retrieval_node and recursive_retrieval_node apply rerank_k cutoff.

Bug: both nodes called voyage_rerank with top_k=len(all_chunks), discarding
rerank_k from compute_dynamic_k entirely.
"""
from unittest.mock import call, patch

import config


# ---------------------------------------------------------------------------
# retrieval_node — hybrid_course
# ---------------------------------------------------------------------------

def test_hybrid_course_rerank_uses_rerank_k_not_all_chunks():
    """voyage_rerank must be called with rerank_k, not len(all_chunks)."""
    from rag.nodes.retrieval_node import retrieval_node

    # 1 course → rerank_k = config.PER_COURSE_K["hybrid_course"]["rerank_k"] * 1
    expected_rerank_k = config.PER_COURSE_K["hybrid_course"]["rerank_k"]
    # Return more chunks than rerank_k to expose the bug
    fake_chunks = [{"content": f"chunk {i}", "score": 0.9} for i in range(20)]

    state = {
        "function": "hybrid_course",
        "course_ids": ["202611_CSCE_221_500"],
        "rewritten_query": "what is the syllabus?",
        "node_trace": [],
        "retrieval_cache": {},
    }

    with patch("rag.tools.mongo.hybrid_search", return_value=fake_chunks), \
         patch("rag.tools.voyage.rerank", return_value=fake_chunks[:expected_rerank_k]) as mock_rerank:
        retrieval_node(state)

    mock_rerank.assert_called_once()
    called_top_k = mock_rerank.call_args[1]["top_k"]
    assert called_top_k == expected_rerank_k, (
        f"Expected top_k={expected_rerank_k} (rerank_k), got top_k={called_top_k} (len of all chunks)"
    )


# ---------------------------------------------------------------------------
# retrieval_node — semantic_general
# ---------------------------------------------------------------------------

def test_semantic_general_rerank_uses_rerank_k_not_all_chunks():
    """semantic_general must also apply rerank_k cutoff."""
    from rag.nodes.retrieval_node import retrieval_node

    expected_rerank_k = config.PER_COURSE_K["semantic_general"]["rerank_k"]
    fake_chunks = [{"content": f"chunk {i}", "score": 0.9} for i in range(20)]

    state = {
        "function": "semantic_general",
        "course_ids": [],
        "rewritten_query": "ML courses at TAMU",
        "node_trace": [],
        "retrieval_cache": {},
    }

    with patch("rag.tools.mongo.semantic_search", return_value=fake_chunks), \
         patch("rag.tools.voyage.rerank", return_value=fake_chunks[:expected_rerank_k]) as mock_rerank:
        retrieval_node(state)

    mock_rerank.assert_called_once()
    called_top_k = mock_rerank.call_args[1]["top_k"]
    assert called_top_k == expected_rerank_k, (
        f"Expected top_k={expected_rerank_k} (rerank_k), got top_k={called_top_k}"
    )


# ---------------------------------------------------------------------------
# recursive_retrieval_node
# ---------------------------------------------------------------------------

def test_recursive_retrieval_rerank_uses_rerank_k_not_all_chunks():
    """recursive_retrieval_node must apply rerank_k cutoff."""
    from rag.nodes.recursive_retrieval_node import recursive_retrieval_node

    expected_rerank_k = config.PER_COURSE_K["recursive"]["rerank_k"]
    fake_chunks = [{"content": f"chunk {i}", "score": 0.9} for i in range(20)]

    state = {
        "function": "recursive",
        "course_ids": ["202611_CSCE_221_500"],
        "rewritten_query": "prereqs for CSCE 221",
        "node_trace": [],
        "retrieval_cache": {},
    }

    with patch("rag.tools.mongo.hybrid_search", return_value=fake_chunks), \
         patch("rag.tools.voyage.rerank", return_value=fake_chunks[:expected_rerank_k]) as mock_rerank:
        recursive_retrieval_node(state)

    mock_rerank.assert_called_once()
    called_top_k = mock_rerank.call_args[1]["top_k"]
    assert called_top_k == expected_rerank_k, (
        f"Expected top_k={expected_rerank_k} (rerank_k), got top_k={called_top_k}"
    )
