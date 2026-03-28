"""Tests for mem0 integration and session cache (router, retrieval, answer caches)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# cache_utils
# ---------------------------------------------------------------------------

def test_normalize_query_lowercases_and_strips_punctuation():
    from rag.v4.cache_utils import normalize_query
    assert normalize_query("CSCE 221??") == "csce 221"


def test_normalize_query_collapses_whitespace():
    from rag.v4.cache_utils import normalize_query
    assert normalize_query("  what   are  the  prereqs?  ") == "what are the prereqs"


def test_normalize_query_identical_paraphrases_differ():
    """Two differently-phrased queries should NOT produce the same key (exact-match only)."""
    from rag.v4.cache_utils import normalize_query
    assert normalize_query("CSCE 221 grading") != normalize_query("CSCE 221 grading policy")


def test_normalize_query_same_text_same_key():
    from rag.v4.cache_utils import normalize_query
    q = "What is the final exam date?"
    assert normalize_query(q) == normalize_query(q)


# ---------------------------------------------------------------------------
# router_node — cache hit
# ---------------------------------------------------------------------------

def _base_router_state(**extra):
    return {"query": "CSCE 221 grading?", "node_trace": [], "timing_ms": {}, **extra}


def _make_registry_with_router(function="hybrid_course"):
    from rag.router import RouterResult
    registry = MagicMock()
    rr = RouterResult(
        course_ids=["202611_CSCE_221_500"],
        rewritten_query="CSCE 221 grading policy",
        function=function,
        intent_type="ACADEMIC",
    )
    registry.router_llm.classify.return_value = rr
    return registry


def test_router_cache_hit_skips_llm():
    """When router_cache contains the query key, classify() is never called."""
    from rag.v4.cache_utils import normalize_query
    from rag.v4.nodes.router_node import router_node

    registry = _make_registry_with_router()
    cached_entry = {
        "function": "hybrid_course",
        "course_ids": ["202611_CSCE_221_500"],
        "rewritten_query": "CSCE 221 grading policy",
        "intent_type": "ACADEMIC",
        "specific_categories": [],
        "recurrent_search": False,
        "requires_retrieval": True,
    }
    query = "CSCE 221 grading?"
    state = _base_router_state(router_cache={normalize_query(query): cached_entry})

    with patch("config.SESSION_CACHE_ENABLED", True):
        result = router_node(state, registry=registry)

    registry.router_llm.classify.assert_not_called()
    assert "router_cache_hit" in result["node_trace"]
    assert result["function"] == "hybrid_course"
    assert result["course_ids"] == ["202611_CSCE_221_500"]


def test_router_cache_miss_calls_llm_and_writes_cache():
    """On cache miss, classify() is called and the result is written to router_cache."""
    from rag.v4.cache_utils import normalize_query
    from rag.v4.nodes.router_node import router_node

    registry = _make_registry_with_router()
    state = _base_router_state(router_cache={})

    with patch("config.SESSION_CACHE_ENABLED", True):
        result = router_node(state, registry=registry)

    registry.router_llm.classify.assert_called_once()
    assert "router_cache_hit" not in result["node_trace"]
    assert "router" in result["node_trace"]

    written_cache = result.get("router_cache", {})
    cache_key = normalize_query("CSCE 221 grading?")
    assert cache_key in written_cache
    assert written_cache[cache_key]["function"] == "hybrid_course"


def test_router_cache_disabled_does_not_use_cache():
    """When SESSION_CACHE_ENABLED=False, no cache check or write happens."""
    from rag.v4.nodes.router_node import router_node

    registry = _make_registry_with_router()
    # Pre-populate cache — should be ignored
    state = _base_router_state(router_cache={"csce 221 grading": {"function": "out_of_scope", "course_ids": []}})

    with patch("config.SESSION_CACHE_ENABLED", False):
        result = router_node(state, registry=registry)

    # LLM must have been called because cache was disabled
    registry.router_llm.classify.assert_called_once()
    assert "router_cache_hit" not in result["node_trace"]
    # No router_cache written to result
    assert not result.get("router_cache")


# ---------------------------------------------------------------------------
# retrieval_node — cache
# ---------------------------------------------------------------------------

def _base_retrieval_state(function="hybrid_course", **extra):
    return {
        "function": function,
        "course_ids": ["202611_CSCE_221_500"],
        "rewritten_query": "CSCE 221 grading policy",
        "eval_query": "",
        "specific_categories": [],
        "node_trace": [],
        "timing_ms": {},
        **extra,
    }


def _make_retrieval_registry(chunks=None):
    registry = MagicMock()
    if chunks is None:
        chunks = [{"content": "chunk text", "score": 0.9}]
    registry.retriever.hybrid_search.return_value = chunks
    registry.retriever.semantic_search.return_value = chunks
    registry.reranker.rerank.return_value = chunks
    return registry


def test_retrieval_cache_hit_skips_retrieval_hybrid():
    """On cache hit for hybrid_course, retriever and reranker are never called."""
    from rag.v4.nodes.retrieval_node import _make_retrieval_cache_key, retrieval_node

    fake_chunks = [{"content": "cached chunk", "score": 0.95}]
    cache_key = _make_retrieval_cache_key(
        "hybrid_course", ["202611_CSCE_221_500"], "CSCE 221 grading policy", ""
    )
    state = _base_retrieval_state(retrieval_cache={cache_key: fake_chunks})
    registry = _make_retrieval_registry()

    with patch("config.SESSION_CACHE_ENABLED", True):
        result = retrieval_node(state, registry=registry)

    registry.retriever.hybrid_search.assert_not_called()
    registry.reranker.rerank.assert_not_called()
    assert "retrieval_cache_hit" in result["node_trace"]
    assert result["retrieved_chunks"] == fake_chunks


def test_retrieval_cache_hit_skips_retrieval_semantic():
    """On cache hit for semantic_general, retriever and reranker are never called."""
    from rag.v4.nodes.retrieval_node import _make_retrieval_cache_key, retrieval_node

    fake_chunks = [{"content": "semantic cached", "score": 0.8}]
    cache_key = _make_retrieval_cache_key(
        "semantic_general", [], "CSCE 221 grading policy", ""
    )
    state = _base_retrieval_state(
        function="semantic_general",
        course_ids=[],
        retrieval_cache={cache_key: fake_chunks},
    )
    registry = _make_retrieval_registry()

    with patch("config.SESSION_CACHE_ENABLED", True):
        result = retrieval_node(state, registry=registry)

    registry.retriever.semantic_search.assert_not_called()
    registry.reranker.rerank.assert_not_called()
    assert "retrieval_cache_hit" in result["node_trace"]


def test_retrieval_cache_miss_calls_retrieval_and_writes_cache():
    """On cache miss for hybrid_course, retrieval is executed and written to retrieval_cache."""
    from rag.v4.nodes.retrieval_node import _make_retrieval_cache_key, retrieval_node

    registry = _make_retrieval_registry()
    state = _base_retrieval_state(retrieval_cache={})

    with patch("config.SESSION_CACHE_ENABLED", True):
        result = retrieval_node(state, registry=registry)

    registry.retriever.hybrid_search.assert_called_once()
    assert "retrieval_cache_hit" not in result["node_trace"]

    written_cache = result.get("retrieval_cache", {})
    cache_key = _make_retrieval_cache_key(
        "hybrid_course", ["202611_CSCE_221_500"], "CSCE 221 grading policy", ""
    )
    assert cache_key in written_cache


def test_retrieval_cache_key_recurrent_uses_eval_query():
    """Recurrent path cache key is based on eval_query, not rewritten_query."""
    from rag.v4.nodes.retrieval_node import _make_retrieval_cache_key

    key1 = _make_retrieval_cache_key("recurrent", ["cid1"], "rewritten", "eval query abc")
    key2 = _make_retrieval_cache_key("recurrent", ["cid1"], "other rewritten", "eval query abc")
    assert key1 == key2  # same eval_query → same key despite different rewritten_query
    assert key1.startswith("recurrent|")


# ---------------------------------------------------------------------------
# history_update_node — answer cache
# ---------------------------------------------------------------------------

def _base_update_state(**extra):
    return {
        "query": "CSCE 221 grading?",
        "answer": "The course uses 40% exams.",
        "history": [],
        "router_result": None,
        "turn_number": 0,
        "node_trace": [],
        "timing_ms": {},
        **extra,
    }


def test_history_update_writes_answer_cache():
    """history_update_node should write normalize(query) → answer to answer_cache."""
    from rag.v4.cache_utils import normalize_query
    from rag.v4.nodes.history_update_node import history_update_node

    state = _base_update_state()

    with patch("config.SESSION_CACHE_ENABLED", True), \
         patch("config.MEM0_ENABLED", False):
        result = history_update_node(state, registry=MagicMock())

    cache = result.get("answer_cache", {})
    key = normalize_query("CSCE 221 grading?")
    assert key in cache
    assert cache[key] == "The course uses 40% exams."


def test_history_update_answer_cache_merges_with_existing():
    """answer_cache should merge new entry with any pre-existing entries."""
    from rag.v4.cache_utils import normalize_query
    from rag.v4.nodes.history_update_node import history_update_node

    existing = {normalize_query("old question"): "old answer"}
    state = _base_update_state(answer_cache=existing)

    with patch("config.SESSION_CACHE_ENABLED", True), \
         patch("config.MEM0_ENABLED", False):
        result = history_update_node(state, registry=MagicMock())

    cache = result.get("answer_cache", {})
    assert normalize_query("old question") in cache
    assert normalize_query("CSCE 221 grading?") in cache


def test_history_update_no_answer_cache_when_disabled():
    """When SESSION_CACHE_ENABLED=False, no answer_cache entry is written."""
    from rag.v4.nodes.history_update_node import history_update_node

    state = _base_update_state()

    with patch("config.SESSION_CACHE_ENABLED", False), \
         patch("config.MEM0_ENABLED", False):
        result = history_update_node(state, registry=MagicMock())

    cache = result.get("answer_cache", {})
    assert len(cache) == 0


def test_history_update_no_answer_cache_when_query_empty():
    """No cache write when query is empty."""
    from rag.v4.nodes.history_update_node import history_update_node

    state = _base_update_state(query="", answer="some answer")

    with patch("config.SESSION_CACHE_ENABLED", True), \
         patch("config.MEM0_ENABLED", False):
        result = history_update_node(state, registry=MagicMock())

    assert len(result.get("answer_cache", {})) == 0


# ---------------------------------------------------------------------------
# history_inject_node — mem0 context path
# ---------------------------------------------------------------------------

def test_history_inject_uses_mem0_context_when_available():
    """When mem0 returns facts, enriched_query uses [Context:] block, not raw history."""
    from rag.v4.nodes.history_inject_node import history_inject_node

    mock_manager = MagicMock()
    mock_manager.search_context.return_value = "- Student studies CSCE 221\n- Interested in grading"

    state = {
        "query": "what is the grading?",
        "rewritten_query": "what is the grading?",
        "session_id": "test-session-123",
        "history": [
            {"role": "user", "content": "Tell me about CSCE 221"},
            {"role": "assistant", "content": "It covers data structures."},
        ],
        "node_trace": [],
        "timing_ms": {},
    }

    with patch("config.MEM0_ENABLED", True), \
         patch("rag.v4.mem0_registry.get", return_value=mock_manager):
        result = history_inject_node(state, registry=MagicMock())

    new_query = result.get("rewritten_query", state["rewritten_query"])
    assert "[Context:" in new_query
    assert "Student studies CSCE 221" in new_query
    assert "Current question:" in new_query
    # Should NOT fall through to raw history
    assert "[Previous context:" not in new_query


def test_history_inject_falls_back_to_raw_history_when_mem0_empty():
    """When mem0 returns empty string, history_inject falls through to raw history logic."""
    from rag.v4.nodes.history_inject_node import history_inject_node

    mock_manager = MagicMock()
    mock_manager.search_context.return_value = ""  # empty → fall through

    state = {
        "query": "prerequisites?",
        "rewritten_query": "prerequisites?",
        "session_id": "test-session-456",
        "history": [
            {"role": "user", "content": "Tell me about CSCE 221"},
            {"role": "assistant", "content": "CSCE 221 is a data structures course."},
        ],
        "node_trace": [],
        "timing_ms": {},
    }

    with patch("config.MEM0_ENABLED", True), \
         patch("rag.v4.mem0_registry.get", return_value=mock_manager):
        result = history_inject_node(state, registry=MagicMock())

    new_query = result.get("rewritten_query", state["rewritten_query"])
    # Should fall through to raw history path
    assert "[Previous context:" in new_query
    assert "CSCE 221" in new_query


def test_history_inject_skips_mem0_when_disabled():
    """When MEM0_ENABLED=False, mem0_registry is never accessed."""
    from rag.v4.nodes.history_inject_node import history_inject_node

    state = {
        "query": "office hours?",
        "rewritten_query": "office hours?",
        "session_id": "test-session-789",
        "history": [
            {"role": "user", "content": "Tell me about CSCE 221"},
            {"role": "assistant", "content": "Data structures course."},
        ],
        "node_trace": [],
        "timing_ms": {},
    }

    with patch("config.MEM0_ENABLED", False), \
         patch("rag.v4.mem0_registry.get") as mock_get:
        result = history_inject_node(state, registry=MagicMock())

    mock_get.assert_not_called()
    new_query = result.get("rewritten_query", state["rewritten_query"])
    # Falls through to raw history
    assert "[Previous context:" in new_query


def test_history_inject_skips_mem0_when_no_session_id():
    """When session_id is empty, mem0 path is skipped even if MEM0_ENABLED=True."""
    from rag.v4.nodes.history_inject_node import history_inject_node

    mock_manager = MagicMock()

    state = {
        "query": "office hours?",
        "rewritten_query": "office hours?",
        "session_id": "",  # empty → skip mem0
        "history": [
            {"role": "user", "content": "Tell me about CSCE 221"},
            {"role": "assistant", "content": "Data structures course."},
        ],
        "node_trace": [],
        "timing_ms": {},
    }

    with patch("config.MEM0_ENABLED", True), \
         patch("rag.v4.mem0_registry.get", return_value=mock_manager):
        history_inject_node(state, registry=MagicMock())

    mock_manager.search_context.assert_not_called()


# ---------------------------------------------------------------------------
# mem0_registry
# ---------------------------------------------------------------------------

def test_mem0_registry_register_and_get():
    from rag.v4 import mem0_registry

    mock_mgr = MagicMock()
    mem0_registry.register("session-abc", mock_mgr)
    assert mem0_registry.get("session-abc") is mock_mgr
    # cleanup
    mem0_registry.unregister("session-abc")


def test_mem0_registry_get_missing_returns_none():
    from rag.v4 import mem0_registry
    assert mem0_registry.get("nonexistent-session") is None


def test_mem0_registry_unregister_removes_entry():
    from rag.v4 import mem0_registry

    mock_mgr = MagicMock()
    mem0_registry.register("to-remove", mock_mgr)
    mem0_registry.unregister("to-remove")
    assert mem0_registry.get("to-remove") is None
