"""Tests for history_inject_node and history_update_node."""
from unittest.mock import MagicMock


def test_history_inject_empty_history_unchanged():
    """With no history, rewritten_query is unchanged."""
    from rag.nodes.history_inject_node import history_inject_node
    state = {
        "query": "what is CSCE 221?",
        "rewritten_query": "what is CSCE 221?",
        "history": [],
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state)
    # rewritten_query should not be in result (unchanged) OR be equal to original
    new_query = result.get("rewritten_query", state["rewritten_query"])
    assert "CSCE 221" in new_query


def test_history_inject_with_2_prior_turns_enriches_history_context():
    """With prior history, history_context should include context; rewritten_query unchanged."""
    from rag.nodes.history_inject_node import history_inject_node
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
    result = history_inject_node(state)
    # rewritten_query must be unchanged
    assert result.get("rewritten_query", state["rewritten_query"]) == state["rewritten_query"]
    # prior history should land in history_context
    assert "CSCE 221" in result.get("history_context", "")


def test_history_update_appends_turn():
    """history_update_node should append current query + answer to history."""
    from rag.nodes.history_update_node import history_update_node
    state = {
        "query": "CSCE 221 grading?",
        "answer": "The grading uses 40% exams...",
        "history": [],
        "router_result": None,
        "turn_number": 0,
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_update_node(state)
    history = result.get("history", [])
    assert len(history) == 2  # user + assistant
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "The grading uses 40% exams..."


def test_history_update_increments_turn_number():
    from rag.nodes.history_update_node import history_update_node
    state = {
        "query": "test", "answer": "ok", "history": [],
        "router_result": None, "turn_number": 3,
        "node_trace": [], "timing_ms": {},
    }
    result = history_update_node(state)
    assert result["turn_number"] == 4


def test_history_update_compression_triggered_at_max_turns():
    """When history exceeds MAX_HISTORY_TURNS * 2 messages, compression truncates."""
    import config
    from rag.nodes.history_update_node import history_update_node

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
    result = history_update_node(state)
    # History should be compressed/truncated
    assert len(result["history"]) <= (config.V4_MAX_HISTORY_TURNS * 2) + 2


def test_history_update_clears_non_checkpointable_fields():
    """history_update_node must set trace and answer_stream to None."""
    from rag.nodes.history_update_node import history_update_node
    state = {
        "query": "test", "answer": "ok", "history": [],
        "router_result": None, "turn_number": 0,
        "trace": object(), "answer_stream": iter([]),
        "node_trace": [], "timing_ms": {},
    }
    result = history_update_node(state)
    assert result.get("trace") is None
    assert result.get("answer_stream") is None


def test_history_inject_caps_at_6_messages():
    """history_inject_node only uses last 6 messages (3 turns) of context."""
    from rag.nodes.history_inject_node import history_inject_node

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
    result = history_inject_node(state)
    # rewritten_query must be unchanged (context goes to history_context now)
    assert result.get("rewritten_query", state["rewritten_query"]) == state["rewritten_query"]
    # history_context should contain recent turns but not earliest
    history_ctx = result.get("history_context", "")
    assert "old question 0" not in history_ctx
    assert history_ctx != ""  # some context was captured


def test_history_inject_includes_summary_when_present():
    """history_inject_node includes history_summary in history_context."""
    from rag.nodes.history_inject_node import history_inject_node

    state = {
        "query": "what are office hours?",
        "history": [
            {"role": "user", "content": "Tell me about CSCE 221"},
            {"role": "assistant", "content": "CSCE 221 covers data structures."},
        ],
        "history_summary": "The student asked about CSCE 221 prerequisites and grading.",
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state)
    ctx = result.get("history_context", "")
    assert "The student asked about CSCE 221 prerequisites" in ctx
    assert "CSCE 221" in ctx


def test_history_inject_summary_appears_before_recent_turns():
    """Summary block must appear before recent turns in history_context."""
    from rag.nodes.history_inject_node import history_inject_node

    state = {
        "query": "any updates?",
        "history": [
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ],
        "history_summary": "Summary of older turns.",
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state)
    ctx = result.get("history_context", "")
    summary_pos = ctx.find("Summary of older turns.")
    recent_pos = ctx.find("recent question")
    assert summary_pos != -1, "Summary block not found in history_context"
    assert recent_pos != -1, "Recent turn not found in history_context"
    assert summary_pos < recent_pos, "Summary must appear before recent turns"
    assert summary_pos < recent_pos, "Summary must appear before recent turns"


def test_history_update_stores_router_result_summary():
    """history_update_node stores function and course_ids in rr_summary."""
    from unittest.mock import MagicMock
    from rag.router import RouterResult
    from rag.nodes.history_update_node import history_update_node

    rr = RouterResult(
        course_ids=["CSCE 638"],
        rewritten_query="schedule CSCE 638")
    state = {
        "query": "schedule for CSCE 638?",
        "answer": "MWF 9-10am.",
        "history": [],
        "router_result": rr,
        "turn_number": 0,
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_update_node(state)
    history = result["history"]
    assistant_msg = next(m for m in history if m["role"] == "assistant")
    assert assistant_msg["router_result"]["course_ids"] == ["CSCE 638"]
    assert "function" in assistant_msg["router_result"]
    assert "specific_categories" not in assistant_msg["router_result"]


def test_history_inject_writes_history_context_not_rewritten_query():
    """history_inject_node stores context in history_context; rewritten_query stays clean."""
    from unittest.mock import MagicMock
    from rag.nodes.history_inject_node import history_inject_node

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
    result = history_inject_node(state)

    # rewritten_query must not be modified
    assert result.get("rewritten_query", original_query) == original_query

    # history_context must be set and contain prior turn content
    assert "history_context" in result
    assert "CSCE 638" in result["history_context"]


def test_history_inject_empty_history_no_history_context():
    """With no history, history_inject_node does not set history_context."""
    from unittest.mock import MagicMock
    from rag.nodes.history_inject_node import history_inject_node

    state = {
        "query": "what is CSCE 221?",
        "rewritten_query": "what is CSCE 221?",
        "history": [],
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state)
    assert result.get("history_context", "") == ""
    assert result.get("rewritten_query", state["rewritten_query"]) == state["rewritten_query"]


def test_history_update_node_updates_summary_on_second_turn():
    """history_update_node should call LLM to update history_summary when turn_number > 1."""
    from unittest.mock import MagicMock, patch
    from rag.nodes.history_update_node import history_update_node

    state = {
        "query": "What courses are available?",
        "answer": "There are many CS courses.",
        "history": [],
        "history_summary": "",
        "turn_number": 1,  # after increment becomes 2, so summary should be updated
        "session_id": "test-session",
        "node_trace": [],
        "answer_cache": {},
        "router_result": None,
        "history_compressed": False,
        "answer_stream": None,
    }

    mock_llm_result = MagicMock()
    mock_llm_result.text = "User asked about available courses. Bot listed CS options."

    with patch("rag.nodes.history_update_node.call_llm", return_value=mock_llm_result) as mock_llm, \
         patch("config.MEM0_ENABLED", True), \
         patch("config.SESSION_CACHE_ENABLED", False), \
         patch("config.MEM0_API_KEY", "test-key"), \
         patch("rag.nodes.history_update_node.Mem0Manager") as mock_mem0:
        mock_mem0.return_value.add_turn_async = MagicMock()
        result = history_update_node(state)

    mock_llm.assert_called_once()
    assert result["history_summary"] == "User asked about available courses. Bot listed CS options."


def test_history_update_node_skips_summary_on_first_turn():
    """history_update_node should NOT call LLM on turn_number 0 (first turn)."""
    from unittest.mock import MagicMock, patch
    from rag.nodes.history_update_node import history_update_node

    state = {
        "query": "Hello",
        "answer": "Hi there.",
        "history": [],
        "history_summary": "",
        "turn_number": 0,  # first turn — after increment becomes 1, no summary update
        "session_id": "test-session",
        "node_trace": [],
        "answer_cache": {},
        "router_result": None,
        "history_compressed": False,
        "answer_stream": None,
    }

    with patch("rag.nodes.history_update_node.call_llm") as mock_llm, \
         patch("config.MEM0_ENABLED", True), \
         patch("config.SESSION_CACHE_ENABLED", False), \
         patch("config.MEM0_API_KEY", "test-key"), \
         patch("rag.nodes.history_update_node.Mem0Manager") as mock_mem0:
        mock_mem0.return_value.add_turn_async = MagicMock()
        result = history_update_node(state)

    mock_llm.assert_not_called()
    assert result["history_summary"] == ""


def test_history_inject_node_hybrid_context():
    """history_inject_node should combine facts + gist + last 2 turns."""
    from unittest.mock import MagicMock, patch
    from rag.nodes.history_inject_node import history_inject_node

    state = {
        "query": "Which ML course should I take?",
        "rewritten_query": None,
        # history contains only completed turns (history_inject runs before history_update)
        "history": [
            {"role": "user", "content": "What are the CS electives?"},
            {"role": "assistant", "content": "There are many options including CSCE 478."},
            {"role": "user", "content": "What about data science?"},
            {"role": "assistant", "content": "CSCE 689 covers data science topics."},
        ],
        "history_summary": "User is a CS senior exploring electives. Previously asked about CS electives and data science.",
        "session_id": "test-session",
        "node_trace": [],
    }

    mock_manager = MagicMock()
    mock_manager.search_context.return_value = "- user is a CS senior\n- prefers afternoon classes"

    with patch("rag.nodes.history_inject_node.Mem0Manager", return_value=mock_manager), \
         patch("config.MEM0_ENABLED", True), \
         patch("config.MEM0_API_KEY", "test-key"):
        result = history_inject_node(state)

    ctx = result["history_context"]
    # Facts layer
    assert "CS senior" in ctx
    # Gist layer
    assert "CS senior exploring" in ctx
    # Flow layer — last 2 turns (4 messages) — both turns present
    assert "CSCE 478" in ctx
    assert "CSCE 689" in ctx


def test_history_inject_node_no_mem0_falls_back_to_gist_and_flow():
    """Without mem0 enabled, should still return gist + flow."""
    from unittest.mock import patch
    from rag.nodes.history_inject_node import history_inject_node

    state = {
        "query": "Tell me more",
        "rewritten_query": None,
        "history": [
            {"role": "user", "content": "What is CSCE 411?"},
            {"role": "assistant", "content": "CSCE 411 covers algorithms."},
        ],
        "history_summary": "User asked about CSCE 411.",
        "session_id": "",
        "node_trace": [],
    }

    with patch("config.MEM0_ENABLED", False):
        result = history_inject_node(state)

    ctx = result["history_context"]
    assert "User asked about CSCE 411" in ctx
    assert "algorithms" in ctx
