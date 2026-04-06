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


from rag.nodes.recursive_generator_node import recursive_generator_node


def test_recursive_generator_overwrites_function_and_query():
    state = {
        "query": "what courses should I take with CSCE 605?",
        "recursive_chunks": [{"course_id": "CSCE 605", "text": "graph algorithms and systems"}],
        "course_ids": ["CSCE 605"],
        "intent_type": "ACADEMIC",
        "history_context": "",
        "node_trace": [],
        "timing_ms": {},
    }
    llm_response = '{"function": "semantic_general", "course_ids": [], "rewritten_query": "graduate systems courses complementing graph theory"}'
    with patch("rag.tools.llm.call_llm") as mock_llm:
        mock_llm.return_value.text = llm_response
        result = recursive_generator_node(state)

    assert result["function"] == "semantic_general"
    assert result["course_ids"] == []
    assert result["rewritten_query"] == "graduate systems courses complementing graph theory"
    assert "recursive_generator" in result["node_trace"]


def test_recursive_generator_hybrid_course_output():
    state = {
        "query": "explain the prerequisites for CSCE 605",
        "recursive_chunks": [{"course_id": "CSCE 605", "text": "requires CSCE 520 and CSCE 601"}],
        "course_ids": ["CSCE 605"],
        "intent_type": None,
        "history_context": "",
        "node_trace": [],
        "timing_ms": {},
    }
    llm_response = '{"function": "hybrid_course", "course_ids": ["CSCE 520", "CSCE 601"], "rewritten_query": "prerequisite foundational content"}'
    with patch("rag.tools.llm.call_llm") as mock_llm:
        mock_llm.return_value.text = llm_response
        result = recursive_generator_node(state)

    assert result["function"] == "hybrid_course"
    assert "CSCE 520" in result["course_ids"]
    assert result["rewritten_query"] == "prerequisite foundational content"


def test_recursive_generator_fallback_on_llm_failure():
    state = {
        "query": "what should I take with CSCE 605?",
        "recursive_chunks": [],
        "course_ids": ["CSCE 605"],
        "intent_type": None,
        "history_context": "",
        "node_trace": [],
        "timing_ms": {},
    }
    with patch("rag.tools.llm.call_llm", side_effect=Exception("LLM timeout")):
        result = recursive_generator_node(state)

    assert result["function"] == "semantic_general"
    assert result["course_ids"] == []
    assert result["rewritten_query"] == "what should I take with CSCE 605?"


def test_recursive_generator_rejects_invalid_function():
    state = {
        "query": "what should I take with CSCE 605?",
        "recursive_chunks": [],
        "course_ids": ["CSCE 605"],
        "intent_type": None,
        "history_context": "",
        "node_trace": [],
        "timing_ms": {},
    }
    llm_response = '{"function": "recursive", "course_ids": [], "rewritten_query": "some query"}'
    with patch("rag.tools.llm.call_llm") as mock_llm:
        mock_llm.return_value.text = llm_response
        result = recursive_generator_node(state)

    assert result["function"] == "semantic_general"
