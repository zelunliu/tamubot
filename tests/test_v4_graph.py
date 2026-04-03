"""Tests for the LangGraph graph with mock tools."""
from unittest.mock import MagicMock, patch
from rag.graph.builder import build_graph
from rag.router import RouterResult


def _base_state(**extra):
    return {
        "query": "test query",
        "node_trace": [], "timing_ms": {},
        "conflicted_course_ids": [], "data_gaps": [], "data_integrity": True,
        "anchor_chunks": [], "discovery_chunks": [], "retrieved_chunks": [],
        **extra,
    }


def _invoke_graph(function="hybrid_course", query="what are office hours?"):
    """Invoke graph with mocked tools for the given function."""
    is_recurrent = function == "recurrent"
    rr = RouterResult(
        course_ids=["202611_CSCE_221_500"] if function not in ("out_of_scope", "semantic_general") else [],
        rewritten_query="test query",
        function=function,
        intent_type="ACADEMIC" if function != "out_of_scope" else None,
        recurrent_search=is_recurrent,
    )

    with patch("rag.router.classify_query", return_value=rr), \
         patch("rag.tools.mongo.fetch_anchor_chunks", return_value=(
             [{"course_id": "202611_CSCE_221_500", "chunk_index": 0, "text": "anchor chunk"}], [], True
         )), \
         patch("rag.tools.mongo.hybrid_search", return_value=[
             {"course_id": "202611_CSCE_221_500", "chunk_index": 1, "text": "result"}
         ]), \
         patch("rag.tools.mongo.semantic_search", return_value=[
             {"course_id": "202611_CSCE_314_500", "chunk_index": 0, "text": "sem result"}
         ]), \
         patch("rag.tools.mongo.get_meeting_times", return_value={}), \
         patch("rag.tools.voyage.rerank", side_effect=lambda q, c, top_k: c[:top_k] if c else []), \
         patch("rag.generator.generate_stream", return_value=iter(["Hello ", "world"])), \
         patch("rag.generator.generate_eval_search_string", return_value="expanded query"):
        graph = build_graph()
        return graph.invoke(_base_state(query=query))


def test_hybrid_course_path():
    result = _invoke_graph("hybrid_course")
    assert "hybrid_course" in result["node_trace"] or "retrieval" in result["node_trace"]
    assert "generator" in result["node_trace"]
    assert result["function"] == "hybrid_course"
    assert isinstance(result["answer"], str)


def test_out_of_scope_path():
    result = _invoke_graph("out_of_scope", query="what is the weather?")
    assert "out_of_scope" in result["node_trace"]
    assert "generator" not in result["node_trace"]
    assert result["answer"]  # canned response


def test_semantic_general_path():
    result = _invoke_graph("semantic_general", query="what CSCE courses cover ML?")
    assert "retrieval" in result["node_trace"]
    assert "generator" in result["node_trace"]
    assert result["function"] == "semantic_general"


def test_recurrent_path():
    result = _invoke_graph("recurrent", query="compare CSCE 221 with similar courses")
    assert "anchor" in result["node_trace"]
    assert "eval_search" in result["node_trace"]
    assert "retrieval" in result["node_trace"]
    assert "schedule_filter" in result["node_trace"]
    assert "merge" in result["node_trace"]
    assert "generator" in result["node_trace"]


def test_result_is_v3_compatible():
    """The graph result must contain all 6 fields returned by v3 run_pipeline()."""
    result = _invoke_graph("hybrid_course")
    assert "retrieved_chunks" in result
    assert "router_result" in result
    assert "data_gaps" in result
    assert "data_integrity" in result
    assert "conflicted_course_ids" in result


def test_anchor_node_calls_fetch_anchor_chunks_with_course_ids():
    """anchor_node calls fetch_anchor_chunks with only course_ids."""
    from rag.nodes.anchor_node import anchor_node

    with patch("rag.tools.mongo.fetch_anchor_chunks", return_value=([], [], True)) as mock_fetch:
        state = {
            "course_ids": ["202611_CSCE_221_500"],
            "node_trace": [],
        }
        anchor_node(state)

    mock_fetch.assert_called_once_with(["202611_CSCE_221_500"])


def test_generator_node_answer_stream_is_list():
    """generator_node must return answer_stream as list[str], not a generator."""
    from rag.nodes.generator_node import generator_node

    with patch("rag.generator.generate_stream", return_value=iter(["Hello ", "world"])):
        state = {"query": "test", "node_trace": [], "timing_ms": {}}
        result = generator_node(state)

    assert isinstance(result["answer_stream"], list)
    assert result["answer_stream"] == ["Hello ", "world"]
    assert result["answer"] == "Hello world"


def test_out_of_scope_node_answer_stream_is_list():
    """out_of_scope_node must return answer_stream as list[str]."""
    from rag.nodes.out_of_scope_node import out_of_scope_node

    state = {"query": "test", "node_trace": [], "timing_ms": {}}
    result = out_of_scope_node(state)

    assert isinstance(result["answer_stream"], list)
    assert len(result["answer_stream"]) == 1
    assert "TamuBot" in result["answer_stream"][0]


def test_pipeline_with_memory_returns_six_tuple():
    """run_pipeline_with_memory must return a 6-tuple with answer_tokens as last element."""
    import rag.graph.pipeline as pipeline_mod
    from rag.graph.pipeline import run_pipeline_with_memory

    mock_result = {
        "retrieved_chunks": [],
        "router_result": MagicMock(function="out_of_scope", course_ids=[], requires_retrieval=False),
        "data_gaps": [],
        "data_integrity": True,
        "conflicted_course_ids": [],
        "answer": "Howdy!",
        "function": "out_of_scope",
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
