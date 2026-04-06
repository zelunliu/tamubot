"""Tests for the LangGraph graph with mock tools."""
from unittest.mock import MagicMock, patch
from rag.graph.builder import build_graph
from rag.router import RouterResult


def _base_state(**extra):
    return {
        "query": "test query",
        "node_trace": [], "timing_ms": {},
        "data_gaps": [], "data_integrity": True,
        "recursive_chunks": [], "retrieved_chunks": [],
        **extra,
    }


def _make_rr(function="hybrid_course"):
    is_recursive = function == "recursive"
    return RouterResult(
        course_ids=["202611_CSCE_221_500"] if function not in ("out_of_scope", "semantic_general") else [],
        rewritten_query="retrieve course CSCE 221" if is_recursive else "test query",
        function=function,
        intent_type="ACADEMIC" if function != "out_of_scope" else None,
        recursive_search=is_recursive,
    )


def _invoke_graph(function="hybrid_course", query="what are office hours?"):
    rr = _make_rr(function)
    with patch("rag.router.classify_query", return_value=rr), \
         patch("rag.tools.mongo.hybrid_search", return_value=[
             {"course_id": "202611_CSCE_221_500", "chunk_index": 0, "text": "result"}
         ]), \
         patch("rag.tools.mongo.semantic_search", return_value=[
             {"course_id": "202611_CSCE_314_500", "chunk_index": 0, "text": "sem result"}
         ]), \
         patch("rag.tools.voyage.rerank", side_effect=lambda q, c, top_k: c[:top_k] if c else []), \
         patch("rag.generator.generate_stream", return_value=iter(["Hello ", "world"])), \
         patch("rag.tools.llm.call_llm") as mock_llm:
        mock_llm.return_value.text = '{"function": "semantic_general", "course_ids": [], "rewritten_query": "related courses"}'
        graph = build_graph()
        return graph.invoke(_base_state(query=query))


def test_hybrid_course_path():
    result = _invoke_graph("hybrid_course")
    assert "retrieval" in result["node_trace"]
    assert "generator" in result["node_trace"]
    assert result["function"] == "hybrid_course"
    assert isinstance(result["answer"], str)


def test_out_of_scope_path():
    result = _invoke_graph("out_of_scope", query="what is the weather?")
    assert "out_of_scope" in result["node_trace"]
    assert "generator" not in result["node_trace"]
    assert result["answer"]


def test_semantic_general_path():
    result = _invoke_graph("semantic_general", query="what CSCE courses cover ML?")
    assert "retrieval" in result["node_trace"]
    assert "generator" in result["node_trace"]


def test_recursive_path_node_trace():
    result = _invoke_graph("recursive", query="what should I take with CSCE 221?")
    expected_nodes = ["router", "recursive_retrieval", "recursive_generator", "retrieval", "generator"]
    for node in expected_nodes:
        assert node in result["node_trace"], f"Missing node: {node}"


def test_recursive_path_no_deleted_nodes():
    result = _invoke_graph("recursive", query="what should I take with CSCE 221?")
    for deleted in ["anchor", "eval_search", "schedule_filter", "merge"]:
        assert deleted not in result["node_trace"], f"Deleted node present: {deleted}"


def test_generator_node_answer_stream_is_list():
    from rag.nodes.generator_node import generator_node
    with patch("rag.generator.generate_stream", return_value=iter(["Hello ", "world"])):
        state = {"query": "test", "node_trace": [], "timing_ms": {}}
        result = generator_node(state)
    assert isinstance(result["answer_stream"], list)
    assert result["answer_stream"] == ["Hello ", "world"]
    assert result["answer"] == "Hello world"


def test_out_of_scope_node_answer_stream_is_list():
    from rag.nodes.out_of_scope_node import out_of_scope_node
    state = {"query": "test", "node_trace": [], "timing_ms": {}}
    result = out_of_scope_node(state)
    assert isinstance(result["answer_stream"], list)
    assert len(result["answer_stream"]) == 1
    assert "TamuBot" in result["answer_stream"][0]


def test_pipeline_with_memory_returns_six_tuple():
    import rag.graph.pipeline as pipeline_mod
    from rag.graph.pipeline import run_pipeline_with_memory

    mock_result = {
        "retrieved_chunks": [],
        "function": "out_of_scope",
        "course_ids": [],
        "intent_type": None,
        "recursive_search": False,
        "rewritten_query": "",
        "retrieval_mode": "",
        "data_gaps": [],
        "data_integrity": True,
        "answer": "Howdy!",
    }

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = mock_result
    original = pipeline_mod._memory_graph
    pipeline_mod._memory_graph = mock_graph
    try:
        result = run_pipeline_with_memory("hello", thread_config={"configurable": {"thread_id": "t1"}})
    finally:
        pipeline_mod._memory_graph = original

    assert len(result) == 6
    chunks, rr, gaps, integrity, conflicted, tokens = result
    assert isinstance(tokens, list)
    assert rr.function == "out_of_scope"
