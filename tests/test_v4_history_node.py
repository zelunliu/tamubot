"""Tests for history_inject_node and history_update_node."""
from unittest.mock import MagicMock


def test_history_inject_empty_history_no_rewritten_query():
    """With no history, history_inject does not set rewritten_query (no context to add)."""
    from rag.nodes.history_inject_node import history_inject_node
    state = {
        "query": "what is CSCE 221?",
        "history": [],
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state)
    # No context available — rewritten_query should not be set
    assert result.get("rewritten_query") is None
    assert result.get("history_context", "") == ""


def test_history_inject_with_2_prior_turns_enriches_query():
    """With prior history, history_inject appends context to rewritten_query and history_context."""
    from rag.nodes.history_inject_node import history_inject_node
    state = {
        "query": "what are the prerequisites?",
        "history": [
            {"role": "user", "content": "Tell me about CSCE 221"},
            {"role": "assistant", "content": "CSCE 221 covers data structures..."},
        ],
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state)
    # rewritten_query must include the original query and context
    rq = result.get("rewritten_query", "")
    assert "what are the prerequisites?" in rq
    assert "CSCE 221" in rq
    # prior history should also land in history_context
    assert "CSCE 221" in result.get("history_context", "")


def test_history_update_appends_turn():
    """history_update_node should append current query + answer to history."""
    from unittest.mock import MagicMock, patch
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
    with patch("rag.nodes.history_update_node.call_llm", return_value=MagicMock(text="")), \
         patch("config.SESSION_CACHE_ENABLED", False), \
         patch("config.MEM0_ENABLED", False):
        result = history_update_node(state)
    history = result.get("history", [])
    assert len(history) == 2  # user + assistant
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "The grading uses 40% exams..."


def test_history_update_increments_turn_number():
    from unittest.mock import MagicMock, patch
    from rag.nodes.history_update_node import history_update_node
    state = {
        "query": "test", "answer": "ok", "history": [],
        "router_result": None, "turn_number": 3,
        "node_trace": [], "timing_ms": {},
    }
    with patch("rag.nodes.history_update_node.call_llm", return_value=MagicMock(text="")), \
         patch("config.SESSION_CACHE_ENABLED", False), \
         patch("config.MEM0_ENABLED", False):
        result = history_update_node(state)
    assert result["turn_number"] == 4


def test_history_update_compression_triggered_at_max_turns():
    """When history exceeds MAX_HISTORY_TURNS * 2 messages, compression truncates."""
    import config
    from unittest.mock import MagicMock, patch
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
    with patch("rag.nodes.history_update_node.call_llm", return_value=MagicMock(text="")), \
         patch("config.SESSION_CACHE_ENABLED", False), \
         patch("config.MEM0_ENABLED", False):
        result = history_update_node(state)
    # History should be compressed/truncated
    assert len(result["history"]) <= (config.V4_MAX_HISTORY_TURNS * 2) + 2


def test_history_update_clears_non_checkpointable_fields():
    """history_update_node must set answer_stream to None."""
    from unittest.mock import MagicMock, patch
    from rag.nodes.history_update_node import history_update_node
    state = {
        "query": "test", "answer": "ok", "history": [],
        "router_result": None, "turn_number": 0,
        "answer_stream": iter([]),
        "node_trace": [], "timing_ms": {},
    }
    with patch("rag.nodes.history_update_node.call_llm", return_value=MagicMock(text="")), \
         patch("config.SESSION_CACHE_ENABLED", False), \
         patch("config.MEM0_ENABLED", False):
        result = history_update_node(state)
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
        "history": long_history,
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state)
    # history_context should contain recent turns but not earliest
    history_ctx = result.get("history_context", "")
    assert "old question 0" not in history_ctx
    assert history_ctx != ""  # some context was captured
    # rewritten_query should include the query and recent context (not earliest turns)
    rq = result.get("rewritten_query", "")
    assert "new question" in rq
    assert "old question 0" not in rq


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
    from unittest.mock import MagicMock, patch
    from rag.state.pipeline_state import RouterResult
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
    with patch("rag.nodes.history_update_node.call_llm", return_value=MagicMock(text="")), \
         patch("config.SESSION_CACHE_ENABLED", False), \
         patch("config.MEM0_ENABLED", False):
        result = history_update_node(state)
    history = result["history"]
    assistant_msg = next(m for m in history if m["role"] == "assistant")
    assert assistant_msg["router_result"]["course_ids"] == ["CSCE 638"]
    assert "function" in assistant_msg["router_result"]
    assert "specific_categories" not in assistant_msg["router_result"]


def test_history_inject_writes_context_to_rewritten_query_and_history_context():
    """history_inject_node appends history_context to rewritten_query; both carry prior turn content."""
    from rag.nodes.history_inject_node import history_inject_node

    original_query = "what are the prerequisites?"
    state = {
        "query": original_query,
        "history": [
            {"role": "user", "content": "Tell me about CSCE 638"},
            {"role": "assistant", "content": "CSCE 638 is a graduate ML course."},
        ],
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state)

    # history_context must contain prior turn content
    assert "history_context" in result
    assert "CSCE 638" in result["history_context"]

    # rewritten_query must start with original query and include context
    rq = result.get("rewritten_query", "")
    assert rq.startswith(original_query)
    assert "CSCE 638" in rq


def test_history_inject_empty_history_no_context():
    """With no history, history_inject_node returns empty history_context and no rewritten_query."""
    from rag.nodes.history_inject_node import history_inject_node

    state = {
        "query": "what is CSCE 221?",
        "history": [],
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state)
    assert result.get("history_context", "") == ""
    assert result.get("rewritten_query") is None


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


def test_history_update_node_updates_summary_on_first_turn():
    """history_update_node should call LLM on turn_number 0 (first turn)."""
    from unittest.mock import MagicMock, patch
    from rag.nodes.history_update_node import history_update_node

    state = {
        "query": "Hello",
        "answer": "Hi there.",
        "history": [],
        "history_summary": "",
        "turn_number": 0,
        "session_id": "test-session",
        "node_trace": [],
        "answer_cache": {},
        "router_result": None,
        "history_compressed": False,
        "answer_stream": None,
    }

    mock_llm_result = MagicMock()
    mock_llm_result.text = "User greeted the bot."

    with patch("rag.nodes.history_update_node.call_llm", return_value=mock_llm_result) as mock_llm, \
         patch("config.MEM0_ENABLED", True), \
         patch("config.SESSION_CACHE_ENABLED", False), \
         patch("config.MEM0_API_KEY", "test-key"), \
         patch("rag.nodes.history_update_node.Mem0Manager") as mock_mem0:
        mock_mem0.return_value.add_turn_async = MagicMock()
        result = history_update_node(state)

    mock_llm.assert_called_once()
    assert result["history_summary"] == "User greeted the bot."


def test_history_update_node_updates_summary_even_when_mem0_disabled():
    """history_update_node always updates history_summary regardless of MEM0_ENABLED."""
    from unittest.mock import MagicMock, patch
    from rag.nodes.history_update_node import history_update_node

    state = {
        "query": "Hello",
        "answer": "Hi there.",
        "history": [],
        "history_summary": "",
        "turn_number": 0,
        "session_id": "test-session",
        "node_trace": [],
        "answer_cache": {},
        "router_result": None,
        "history_compressed": False,
        "answer_stream": None,
    }

    mock_llm_result = MagicMock()
    mock_llm_result.text = "User greeted the bot."

    with patch("rag.nodes.history_update_node.call_llm", return_value=mock_llm_result) as mock_llm, \
         patch("config.MEM0_ENABLED", False), \
         patch("config.SESSION_CACHE_ENABLED", False):
        result = history_update_node(state)

    mock_llm.assert_called_once()
    assert result["history_summary"] == "User greeted the bot."


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
