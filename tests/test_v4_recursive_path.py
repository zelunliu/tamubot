"""Tests for the recursive search path."""
from unittest.mock import patch

from rag.nodes.recursive_generator_node import recursive_generator_node
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
    llm_response = (
        '{"function": "semantic_general", "course_ids": [], '
        '"rewritten_query": "graduate systems courses complementing graph theory"}'
    )
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
    llm_response = (
        '{"function": "hybrid_course", "course_ids": ["CSCE 520", "CSCE 601"], '
        '"rewritten_query": "prerequisite foundational content"}'
    )
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


def test_retrieval_node_has_no_recurrent_branch():
    import inspect

    from rag.nodes.retrieval_node import retrieval_node
    source = inspect.getsource(retrieval_node)
    assert "recurrent" not in source


def test_generator_node_uses_recursive_prompt_when_recursive_search_true():
    from rag.nodes.generator_node import generator_node
    captured = {}

    def fake_generate_stream(results, question, function, **kwargs):
        captured["function"] = function
        return iter(["answer"])

    state = {
        "query": "what should I take with CSCE 605?",
        "retrieved_chunks": [{"course_id": "CSCE 314", "text": "data structures"}],
        "function": "semantic_general",
        "recursive_search": True,
        "course_ids": [],
        "node_trace": [],
        "timing_ms": {},
    }
    with patch("rag.generator.generate_stream", side_effect=fake_generate_stream):
        generator_node(state)

    assert captured["function"] == "recursive"


def test_route_after_router_recursive_goes_to_recursive_retrieval():
    from rag.edges.routing import route_after_router
    assert route_after_router({"function": "recursive"}) == "recursive_retrieval"


def test_route_after_router_hybrid_goes_to_retrieval():
    from rag.edges.routing import route_after_router
    assert route_after_router({"function": "hybrid_course"}) == "retrieval"


def test_route_after_router_out_of_scope():
    from rag.edges.routing import route_after_router
    assert route_after_router({"function": "out_of_scope"}) == "out_of_scope"


def test_route_after_retrieval_does_not_exist():
    import rag.edges.routing as routing_mod
    assert not hasattr(routing_mod, "route_after_retrieval")


def test_build_graph_has_recursive_nodes():
    from rag.graph.builder import build_graph
    graph = build_graph()
    node_names = set(graph.nodes.keys())
    assert "recursive_retrieval" in node_names
    assert "recursive_generator" in node_names
    assert "anchor" not in node_names
    assert "eval_search" not in node_names
    assert "schedule_filter" not in node_names
    assert "merge" not in node_names
