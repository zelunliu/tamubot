"""Tests for the v4 LangGraph graph with mock registry."""
from unittest.mock import MagicMock
from rag.v4.graph import build_graph
from rag.v4.interfaces import ComponentRegistry
from rag.router import RouterResult


def _make_mock_registry(function: str = "hybrid_course") -> ComponentRegistry:
    """Build a mock registry returning canned RouterResult."""
    router = MagicMock()
    rr = RouterResult(
        course_ids=["202611_CSCE_221_500"] if function != "semantic_general" else [],
        rewritten_query="test query",
        function=function,
        intent_type="ACADEMIC" if function != "out_of_scope" else None,
    )
    router.classify.return_value = rr

    retriever = MagicMock()
    retriever.fetch_anchor_chunks.return_value = (
        [{"course_id": "202611_CSCE_221_500", "chunk_index": 0, "text": "anchor chunk"}],
        [],
        True,
    )
    retriever.hybrid_search.return_value = [{"course_id": "202611_CSCE_221_500", "chunk_index": 1, "text": "result"}]
    retriever.semantic_search.return_value = [{"course_id": "202611_CSCE_314_500", "chunk_index": 0, "text": "sem result"}]
    retriever.get_meeting_times.return_value = {}

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, chunks, top_k, specific_categories=None: chunks[:top_k] if chunks else []

    generator = MagicMock()
    generator.generate_stream.return_value = iter(["Hello ", "world"])
    generator.generate_eval_query.return_value = "expanded query"

    return ComponentRegistry(
        router_llm=router,
        retriever=retriever,
        reranker=reranker,
        generator_llm=generator,
    )


def test_hybrid_course_path():
    registry = _make_mock_registry("hybrid_course")
    graph = build_graph(registry)
    result = graph.invoke({"query": "what are office hours?", "node_trace": [], "timing_ms": {}})

    assert "hybrid_course" in result["node_trace"] or "retrieval" in result["node_trace"]
    assert "generator" in result["node_trace"]
    assert result["function"] == "hybrid_course"
    assert isinstance(result["answer"], str)


def test_out_of_scope_path():
    registry = _make_mock_registry("out_of_scope")
    graph = build_graph(registry)
    result = graph.invoke({"query": "what is the weather?", "node_trace": [], "timing_ms": {}})

    assert "out_of_scope" in result["node_trace"]
    assert "generator" not in result["node_trace"]
    assert result["answer"]  # canned response


def test_semantic_general_path():
    registry = _make_mock_registry("semantic_general")
    graph = build_graph(registry)
    result = graph.invoke({"query": "what CSCE courses cover ML?", "node_trace": [], "timing_ms": {}})

    assert "retrieval" in result["node_trace"]
    assert "generator" in result["node_trace"]
    assert result["function"] == "semantic_general"


def test_recurrent_path():
    registry = _make_mock_registry("recurrent")
    # Need recurrent=True on RouterResult
    rr = RouterResult(
        course_ids=["202611_CSCE_221_500"],
        rewritten_query="compare with similar courses",
        function="recurrent",
        intent_type="ACADEMIC",
        recurrent_search=True,
    )
    registry.router_llm.classify.return_value = rr

    graph = build_graph(registry)
    result = graph.invoke({
        "query": "compare CSCE 221 with similar courses",
        "node_trace": [],
        "timing_ms": {},
        "conflicted_course_ids": [],
        "data_gaps": [],
        "data_integrity": True,
        "anchor_chunks": [],
        "discovery_chunks": [],
        "retrieved_chunks": [],
    })

    assert "anchor" in result["node_trace"]
    assert "eval_search" in result["node_trace"]
    assert "retrieval" in result["node_trace"]
    assert "schedule_filter" in result["node_trace"]
    assert "merge" in result["node_trace"]
    assert "generator" in result["node_trace"]


def test_result_is_v3_compatible():
    """The graph result must contain all 6 fields returned by v3 run_pipeline()."""
    registry = _make_mock_registry("hybrid_course")
    graph = build_graph(registry)
    result = graph.invoke({
        "query": "test",
        "node_trace": [],
        "timing_ms": {},
        "conflicted_course_ids": [],
        "data_gaps": [],
        "data_integrity": True,
        "anchor_chunks": [],
        "discovery_chunks": [],
        "retrieved_chunks": [],
    })

    # v3 returns: (chunks, router_result, data_gaps, data_integrity, conflicted_ids, timing_ms)
    assert "retrieved_chunks" in result
    assert "router_result" in result
    assert "data_gaps" in result
    assert "data_integrity" in result
    assert "conflicted_course_ids" in result


def test_anchor_node_passes_specific_categories():
    """anchor_node must pass specific_categories from state to fetch_anchor_chunks."""
    from rag.v4.nodes.anchor_node import anchor_node

    retriever = MagicMock()
    retriever.fetch_anchor_chunks.return_value = ([], [], True)
    registry = MagicMock()
    registry.retriever = retriever

    state = {
        "course_ids": ["202611_CSCE_221_500"],
        "specific_categories": ["GRADING", "EXAMS"],
        "node_trace": [],
    }
    anchor_node(state, registry=registry)

    _, call_categories = retriever.fetch_anchor_chunks.call_args[0]
    assert call_categories == ["GRADING", "EXAMS"]


def test_generator_node_answer_stream_is_list():
    """generator_node must return answer_stream as list[str], not a generator."""
    from unittest.mock import MagicMock
    from rag.v4.nodes.generator_node import generator_node

    registry = MagicMock()
    registry.generator_llm.generate_stream.return_value = iter(["Hello ", "world"])

    state = {"query": "test", "node_trace": [], "timing_ms": {}}
    result = generator_node(state, registry=registry)

    assert isinstance(result["answer_stream"], list)
    assert result["answer_stream"] == ["Hello ", "world"]
    assert result["answer"] == "Hello world"


def test_out_of_scope_node_answer_stream_is_list():
    """out_of_scope_node must return answer_stream as list[str]."""
    from unittest.mock import MagicMock
    from rag.v4.nodes.out_of_scope_node import out_of_scope_node

    state = {"query": "test", "node_trace": [], "timing_ms": {}}
    result = out_of_scope_node(state, registry=MagicMock())

    assert isinstance(result["answer_stream"], list)
    assert len(result["answer_stream"]) == 1
    assert "TamuBot" in result["answer_stream"][0]


def test_pipeline_with_memory_returns_six_tuple():
    """run_pipeline_v4_with_memory must return a 6-tuple with answer_tokens as last element."""
    from unittest.mock import MagicMock, patch
    from rag.v4.pipeline_v4 import run_pipeline_v4_with_memory

    mock_result = {
        "retrieved_chunks": [],
        "router_result": MagicMock(function="out_of_scope", course_ids=[], requires_retrieval=False),
        "data_gaps": [],
        "data_integrity": True,
        "conflicted_course_ids": [],
        "answer_stream": ["Howdy!"],
        "function": "out_of_scope",
    }

    with patch("rag.v4.pipeline_v4._memory_graph") as mock_graph:
        mock_graph.invoke.return_value = mock_result
        result = run_pipeline_v4_with_memory("hello", thread_config={"configurable": {"thread_id": "t1"}})

    assert len(result) == 6
    chunks, rr, gaps, integrity, conflicted, tokens = result
    assert isinstance(tokens, list)
