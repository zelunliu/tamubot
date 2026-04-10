"""Build the RAG LangGraph state machine."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag.edges.routing import route_after_router
from rag.nodes.generator_node import generator_node
from rag.nodes.out_of_scope_node import out_of_scope_node
from rag.nodes.recursive_generator_node import recursive_generator_node
from rag.nodes.recursive_retrieval_node import recursive_retrieval_node
from rag.nodes.retrieval_node import retrieval_node
from rag.nodes.router_node import router_node
from rag.state.pipeline_state import PipelineState


def build_graph():
    """Build and compile the RAG pipeline graph (stateless, no conversation memory)."""
    graph = StateGraph(PipelineState)

    graph.add_node("router", router_node)
    graph.add_node("recursive_retrieval", recursive_retrieval_node)
    graph.add_node("recursive_generator", recursive_generator_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("generator", generator_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "out_of_scope": "out_of_scope",
            "recursive_retrieval": "recursive_retrieval",
            "retrieval": "retrieval",
        },
    )

    graph.add_edge("recursive_retrieval", "recursive_generator")
    graph.add_edge("recursive_generator", "retrieval")
    graph.add_edge("retrieval", "generator")
    graph.add_edge("generator", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile()


def build_graph_eval():
    """Build eval-only graph: router + retrieval, no generator.

    Identical edge structure to build_graph() but terminates at retrieval.
    Use with run_pipeline_eval() to get the same node-level Langfuse traces
    as production runs without running the generator.
    """
    graph = StateGraph(PipelineState)

    graph.add_node("router", router_node)
    graph.add_node("recursive_retrieval", recursive_retrieval_node)
    graph.add_node("recursive_generator", recursive_generator_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "out_of_scope": "out_of_scope",
            "recursive_retrieval": "recursive_retrieval",
            "retrieval": "retrieval",
        },
    )

    graph.add_edge("recursive_retrieval", "recursive_generator")
    graph.add_edge("recursive_generator", "retrieval")
    graph.add_edge("retrieval", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile()


def build_graph_with_memory(checkpointer=None):
    """Build the RAG pipeline graph with conversation memory support."""
    from rag.nodes.history_inject_node import history_inject_node
    from rag.nodes.history_update_node import history_update_node

    graph = StateGraph(PipelineState)

    graph.add_node("history_inject", history_inject_node)
    graph.add_node("router", router_node)
    graph.add_node("recursive_retrieval", recursive_retrieval_node)
    graph.add_node("recursive_generator", recursive_generator_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("generator", generator_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("history_update", history_update_node)

    graph.set_entry_point("history_inject")
    graph.add_edge("history_inject", "router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "out_of_scope": "out_of_scope",
            "recursive_retrieval": "recursive_retrieval",
            "retrieval": "retrieval",
        },
    )

    graph.add_edge("recursive_retrieval", "recursive_generator")
    graph.add_edge("recursive_generator", "retrieval")
    graph.add_edge("retrieval", "generator")
    graph.add_edge("generator", "history_update")
    graph.add_edge("out_of_scope", "history_update")
    graph.add_edge("history_update", END)

    kwargs = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    return graph.compile(**kwargs)
