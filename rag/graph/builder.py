"""Build the RAG LangGraph state machine."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag.edges.routing import route_after_retrieval, route_after_router
from rag.nodes.anchor_node import anchor_node
from rag.nodes.eval_search_node import eval_search_node
from rag.nodes.generator_node import generator_node
from rag.nodes.merge_node import merge_node
from rag.nodes.out_of_scope_node import out_of_scope_node
from rag.nodes.retrieval_node import retrieval_node
from rag.nodes.router_node import router_node
from rag.nodes.schedule_filter_node import schedule_filter_node
from rag.state.pipeline_state import ConversationState, PipelineState


def build_graph():
    """Build and compile the RAG pipeline graph (stateless, no conversation memory)."""
    graph = StateGraph(PipelineState)

    graph.add_node("router", router_node)
    graph.add_node("anchor", anchor_node)
    graph.add_node("eval_search", eval_search_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("schedule_filter", schedule_filter_node)
    graph.add_node("merge", merge_node)
    graph.add_node("generator", generator_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"out_of_scope": "out_of_scope", "anchor": "anchor", "retrieval": "retrieval"},
    )

    graph.add_edge("anchor", "eval_search")
    graph.add_edge("eval_search", "retrieval")

    graph.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {"schedule_filter": "schedule_filter", "generator": "generator"},
    )

    graph.add_edge("schedule_filter", "merge")
    graph.add_edge("merge", "generator")
    graph.add_edge("generator", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile()


def build_graph_with_memory(checkpointer=None):
    """Build the RAG pipeline graph with conversation memory support."""
    from rag.nodes.history_inject_node import history_inject_node
    from rag.nodes.history_update_node import history_update_node

    graph = StateGraph(ConversationState)

    graph.add_node("history_inject", history_inject_node)
    graph.add_node("router", router_node)
    graph.add_node("anchor", anchor_node)
    graph.add_node("eval_search", eval_search_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("schedule_filter", schedule_filter_node)
    graph.add_node("merge", merge_node)
    graph.add_node("generator", generator_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("history_update", history_update_node)

    graph.set_entry_point("history_inject")
    graph.add_edge("history_inject", "router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"out_of_scope": "out_of_scope", "anchor": "anchor", "retrieval": "retrieval"},
    )

    graph.add_edge("anchor", "eval_search")
    graph.add_edge("eval_search", "retrieval")

    graph.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {"schedule_filter": "schedule_filter", "generator": "generator"},
    )

    graph.add_edge("schedule_filter", "merge")
    graph.add_edge("merge", "generator")
    graph.add_edge("generator", "history_update")
    graph.add_edge("out_of_scope", "history_update")
    graph.add_edge("history_update", END)

    kwargs = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    return graph.compile(**kwargs)
