"""Tests for v4 PipelineState contract."""
import pickle

from rag.state.pipeline_state import PipelineState, ConversationMessage


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


def test_router_result_is_picklable():
    """RouterResult dataclass must be picklable for LangGraph checkpointing."""
    from rag.router import RouterResult
    rr = RouterResult(
        course_ids=["202611_CSCE_120_500"],
        rewritten_query="office hours CSCE 120",
        function="hybrid_course",
    )
    pickled = pickle.dumps(rr)
    restored = pickle.loads(pickled)
    assert restored.course_ids == rr.course_ids
    assert restored.function == rr.function


def test_conversation_message_importable():
    msg: ConversationMessage = {"role": "user", "content": "hello"}
    assert msg["role"] == "user"
