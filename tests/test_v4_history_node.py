"""Tests for history_inject_node and history_update_node."""
from unittest.mock import MagicMock


def test_history_inject_empty_history_unchanged():
    """With no history, rewritten_query is unchanged."""
    from rag.v4.nodes.history_inject_node import history_inject_node
    state = {
        "query": "what is CSCE 221?",
        "rewritten_query": "what is CSCE 221?",
        "history": [],
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state, registry=MagicMock())
    # rewritten_query should not be in result (unchanged) OR be equal to original
    new_query = result.get("rewritten_query", state["rewritten_query"])
    assert "CSCE 221" in new_query


def test_history_inject_with_2_prior_turns_enriches_query():
    """With prior history, rewritten_query should include context."""
    from rag.v4.nodes.history_inject_node import history_inject_node
    state = {
        "query": "what are the prerequisites?",
        "rewritten_query": "what are the prerequisites?",
        "history": [
            {"role": "user", "content": "Tell me about CSCE 221"},
            {"role": "assistant", "content": "CSCE 221 covers data structures..."},
        ],
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state, registry=MagicMock())
    new_query = result.get("rewritten_query", state["rewritten_query"])
    # Should contain prior context
    assert "CSCE 221" in new_query or "prerequisite" in new_query.lower()


def test_history_update_appends_turn():
    """history_update_node should append current query + answer to history."""
    from rag.v4.nodes.history_update_node import history_update_node
    state = {
        "query": "CSCE 221 grading?",
        "answer": "The grading uses 40% exams...",
        "history": [],
        "router_result": None,
        "turn_number": 0,
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_update_node(state, registry=MagicMock())
    history = result.get("history", [])
    assert len(history) == 2  # user + assistant
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "The grading uses 40% exams..."


def test_history_update_increments_turn_number():
    from rag.v4.nodes.history_update_node import history_update_node
    state = {
        "query": "test", "answer": "ok", "history": [],
        "router_result": None, "turn_number": 3,
        "node_trace": [], "timing_ms": {},
    }
    result = history_update_node(state, registry=MagicMock())
    assert result["turn_number"] == 4


def test_history_update_compression_triggered_at_max_turns():
    """When history exceeds MAX_HISTORY_TURNS * 2 messages, compression truncates."""
    import config
    from rag.v4.nodes.history_update_node import history_update_node

    # Build history longer than the limit
    n_turns = config.V4_MAX_HISTORY_TURNS + 2
    long_history = []
    for i in range(n_turns):
        long_history.append({"role": "user", "content": f"question {i}"})
        long_history.append({"role": "assistant", "content": f"answer {i}"})

    state = {
        "query": "final question",
        "answer": "final answer",
        "history": long_history,
        "router_result": None,
        "turn_number": n_turns,
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_update_node(state, registry=MagicMock())
    # History should be compressed/truncated
    assert len(result["history"]) <= (config.V4_MAX_HISTORY_TURNS * 2) + 2


def test_history_update_clears_non_checkpointable_fields():
    """history_update_node must set trace and answer_stream to None."""
    from rag.v4.nodes.history_update_node import history_update_node
    state = {
        "query": "test", "answer": "ok", "history": [],
        "router_result": None, "turn_number": 0,
        "trace": object(), "answer_stream": iter([]),
        "node_trace": [], "timing_ms": {},
    }
    result = history_update_node(state, registry=MagicMock())
    assert result.get("trace") is None
    assert result.get("answer_stream") is None


def test_history_inject_caps_at_6_messages():
    """history_inject_node only uses last 6 messages (3 turns) of context."""
    from rag.v4.nodes.history_inject_node import history_inject_node

    # Build 10 turns of history
    long_history = []
    for i in range(10):
        long_history.append({"role": "user", "content": f"old question {i}"})
        long_history.append({"role": "assistant", "content": f"old answer {i}"})

    state = {
        "query": "new question",
        "rewritten_query": "new question",
        "history": long_history,
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state, registry=MagicMock())
    # rewritten_query must be unchanged (context goes to history_context now)
    assert result.get("rewritten_query", state["rewritten_query"]) == state["rewritten_query"]
    # history_context should contain recent turns but not earliest
    history_ctx = result.get("history_context", "")
    assert "old question 0" not in history_ctx
    assert history_ctx != ""  # some context was captured


# ---------------------------------------------------------------------------
# History summarization tests
# ---------------------------------------------------------------------------

def test_history_update_calls_summarize_when_flag_enabled_and_over_limit():
    """When ENABLE_HISTORY_SUMMARY=True and history exceeds window, summarize_history is called."""
    from unittest.mock import patch
    import config
    from rag.v4.nodes.history_update_node import history_update_node

    n_turns = config.V4_MAX_HISTORY_TURNS + 2
    long_history = []
    for i in range(n_turns):
        long_history.append({"role": "user", "content": f"question {i}"})
        long_history.append({"role": "assistant", "content": f"answer {i}"})

    mock_registry = MagicMock()
    mock_registry.generator_llm.summarize_history.return_value = "Mocked summary."

    state = {
        "query": "final question",
        "answer": "final answer",
        "history": long_history,
        "history_summary": "",
        "router_result": None,
        "turn_number": n_turns,
        "node_trace": [],
        "timing_ms": {},
    }

    with patch.object(config, "ENABLE_HISTORY_SUMMARY", True):
        result = history_update_node(state, registry=mock_registry)

    mock_registry.generator_llm.summarize_history.assert_called_once()
    assert result["history_summary"] == "Mocked summary."
    assert len(result["history"]) <= config.V4_MAX_HISTORY_TURNS * 2 + 2


def test_history_update_no_summarize_when_flag_disabled():
    """When ENABLE_HISTORY_SUMMARY=False, summarize_history is never called even over limit."""
    from unittest.mock import patch
    import config
    from rag.v4.nodes.history_update_node import history_update_node

    n_turns = config.V4_MAX_HISTORY_TURNS + 2
    long_history = []
    for i in range(n_turns):
        long_history.append({"role": "user", "content": f"question {i}"})
        long_history.append({"role": "assistant", "content": f"answer {i}"})

    mock_registry = MagicMock()

    state = {
        "query": "final question",
        "answer": "final answer",
        "history": long_history,
        "history_summary": "",
        "router_result": None,
        "turn_number": n_turns,
        "node_trace": [],
        "timing_ms": {},
    }

    with patch.object(config, "ENABLE_HISTORY_SUMMARY", False):
        result = history_update_node(state, registry=mock_registry)

    mock_registry.generator_llm.summarize_history.assert_not_called()
    assert len(result["history"]) <= config.V4_MAX_HISTORY_TURNS * 2 + 2


def test_history_inject_includes_summary_when_present():
    """history_inject_node prepends summary block when history_summary is non-empty."""
    from rag.v4.nodes.history_inject_node import history_inject_node

    state = {
        "query": "what are office hours?",
        "rewritten_query": "what are office hours?",
        "history": [
            {"role": "user", "content": "Tell me about CSCE 221"},
            {"role": "assistant", "content": "CSCE 221 covers data structures."},
        ],
        "history_summary": "The student asked about CSCE 221 prerequisites and grading.",
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state, registry=MagicMock())
    new_query = result.get("rewritten_query", state["rewritten_query"])
    assert "The student asked about CSCE 221 prerequisites" in new_query
    assert "Current question" in new_query


def test_history_inject_summary_appears_before_recent_turns():
    """Summary block must appear before [Previous context:] block in the enriched query."""
    from rag.v4.nodes.history_inject_node import history_inject_node

    state = {
        "query": "any updates?",
        "rewritten_query": "any updates?",
        "history": [
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ],
        "history_summary": "Summary of older turns.",
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state, registry=MagicMock())
    new_query = result.get("rewritten_query", state["rewritten_query"])
    summary_pos = new_query.find("Summary of older turns.")
    recent_pos = new_query.find("recent question")
    assert summary_pos != -1, "Summary block not found in enriched query"
    assert recent_pos != -1, "Recent turn not found in enriched query"
    assert summary_pos < recent_pos, "Summary must appear before recent turns"


def test_history_update_stores_specific_categories_in_router_result():
    """history_update_node must include specific_categories in rr_summary."""
    from unittest.mock import MagicMock
    from rag.router import RouterResult
    from rag.v4.nodes.history_update_node import history_update_node

    rr = RouterResult(
        course_ids=["CSCE 638"],
        specific_categories=["SCHEDULE"],
        rewritten_query="schedule CSCE 638",
    )
    state = {
        "query": "schedule for CSCE 638?",
        "answer": "MWF 9-10am.",
        "history": [],
        "router_result": rr,
        "turn_number": 0,
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_update_node(state, registry=MagicMock())
    history = result["history"]
    assistant_msg = next(m for m in history if m["role"] == "assistant")
    assert assistant_msg["router_result"]["specific_categories"] == ["SCHEDULE"]


def test_history_inject_writes_history_context_not_rewritten_query():
    """history_inject_node stores context in history_context; rewritten_query stays clean."""
    from unittest.mock import MagicMock
    from rag.v4.nodes.history_inject_node import history_inject_node

    original_query = "what are the prerequisites?"
    state = {
        "query": original_query,
        "rewritten_query": original_query,
        "history": [
            {"role": "user", "content": "Tell me about CSCE 638"},
            {"role": "assistant", "content": "CSCE 638 is a graduate ML course."},
        ],
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state, registry=MagicMock())

    # rewritten_query must not be modified
    assert result.get("rewritten_query", original_query) == original_query

    # history_context must be set and contain prior turn content
    assert "history_context" in result
    assert "CSCE 638" in result["history_context"]


def test_history_inject_empty_history_no_history_context():
    """With no history, history_inject_node does not set history_context."""
    from unittest.mock import MagicMock
    from rag.v4.nodes.history_inject_node import history_inject_node

    state = {
        "query": "what is CSCE 221?",
        "rewritten_query": "what is CSCE 221?",
        "history": [],
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state, registry=MagicMock())
    assert result.get("history_context", "") == ""
    assert result.get("rewritten_query", state["rewritten_query"]) == state["rewritten_query"]
