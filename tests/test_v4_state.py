"""Tests for v4 PipelineState contract."""
import typing

from rag.state.pipeline_state import ConversationMessage, ConversationState, PipelineState


def test_pipeline_state_importable():
    state: PipelineState = {}
    assert isinstance(state, dict)


def test_pipeline_state_minimal_fields():
    state: PipelineState = {
        "query": "what are office hours for CSCE 120?",
        "node_trace": [],
        "timing_ms": {},
    }
    assert state["query"] == "what are office hours for CSCE 120?"
    assert state["node_trace"] == []


def test_conversation_state_is_alias_for_pipeline_state():
    """ConversationState is now an alias — both refer to the same TypedDict."""
    assert ConversationState is PipelineState


def test_pipeline_state_has_recursive_fields():
    state: PipelineState = {
        "recursive_search": True,
        "recursive_chunks": [{"course_id": "CSCE 605", "text": "something"}],
    }
    assert state["recursive_search"] is True
    assert len(state["recursive_chunks"]) == 1


def test_pipeline_state_has_session_fields():
    """Session fields (formerly ConversationState) live directly in PipelineState."""
    state: PipelineState = {
        "session_id": "abc123",
        "history": [{"role": "user", "content": "hello"}],
        "turn_number": 1,
    }
    assert state["session_id"] == "abc123"
    assert state["turn_number"] == 1


def test_conversation_message_importable():
    msg: ConversationMessage = {"role": "user", "content": "hello"}
    assert msg["role"] == "user"


def test_no_router_result_field_in_pipeline_state():
    """router_result is no longer a field in PipelineState — fields are promoted."""
    hints = typing.get_type_hints(PipelineState)
    assert "router_result" not in hints


def test_recursive_prompt_key_exists():
    from rag.prompts import _FUNCTION_PROMPTS, _FUNCTION_TEMPERATURES
    assert "recursive" in _FUNCTION_PROMPTS
    assert "recurrent" not in _FUNCTION_PROMPTS
    assert "recursive" in _FUNCTION_TEMPERATURES
    assert "recurrent" not in _FUNCTION_TEMPERATURES


def test_router_prompt_uses_recursive_search():
    from rag.prompts import ROUTER_PROMPT
    assert "recursive_search" in ROUTER_PROMPT
    assert "recurrent_search" not in ROUTER_PROMPT
