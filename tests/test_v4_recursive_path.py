"""Tests for the recursive search path."""
from unittest.mock import patch
from rag.nodes.recursive_retrieval_node import recursive_retrieval_node


def test_recursive_retrieval_calls_hybrid_search_per_course():
    chunks = [{"course_id": "CSCE 605", "chunk_index": 0, "text": "graph algorithms"}]
    state = {
        "course_ids": ["CSCE 605"],
        "rewritten_query": "retrieve course CSCE 605",
        "node_trace": [],
        "timing_ms": {},
    }
    with patch("rag.tools.mongo.hybrid_search", return_value=chunks) as mock_hs, \
         patch("rag.tools.voyage.rerank", side_effect=lambda q, c, top_k: c):
        result = recursive_retrieval_node(state)

    mock_hs.assert_called_once()
    assert result["recursive_chunks"] == chunks
    assert "recursive_retrieval" in result["node_trace"]


def test_recursive_retrieval_writes_recursive_chunks_not_retrieved():
    state = {
        "course_ids": ["CSCE 605"],
        "rewritten_query": "retrieve course CSCE 605",
        "node_trace": [],
        "timing_ms": {},
    }
    with patch("rag.tools.mongo.hybrid_search", return_value=[]), \
         patch("rag.tools.voyage.rerank", side_effect=lambda q, c, top_k: c):
        result = recursive_retrieval_node(state)

    assert "recursive_chunks" in result
    assert "retrieved_chunks" not in result


def test_recursive_retrieval_handles_multiple_courses():
    chunk_a = [{"course_id": "CSCE 605", "chunk_index": 0, "text": "a"}]
    chunk_b = [{"course_id": "CSCE 606", "chunk_index": 0, "text": "b"}]
    state = {
        "course_ids": ["CSCE 605", "CSCE 606"],
        "rewritten_query": "retrieve courses CSCE 605 CSCE 606",
        "node_trace": [],
        "timing_ms": {},
    }
    with patch("rag.tools.mongo.hybrid_search", side_effect=[chunk_a, chunk_b]), \
         patch("rag.tools.voyage.rerank", side_effect=lambda q, c, top_k: c):
        result = recursive_retrieval_node(state)

    assert len(result["recursive_chunks"]) == 2


def test_recursive_retrieval_fallback_on_error():
    state = {
        "course_ids": ["CSCE 605"],
        "rewritten_query": "retrieve course CSCE 605",
        "node_trace": [],
        "timing_ms": {},
    }
    with patch("rag.tools.mongo.hybrid_search", side_effect=Exception("DB down")):
        result = recursive_retrieval_node(state)

    assert result["recursive_chunks"] == []
    assert "error" in result
