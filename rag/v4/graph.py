"""Build the v4 LangGraph state machine."""
from __future__ import annotations

import functools

from langgraph.graph import END, StateGraph

from rag.v4.interfaces import ComponentRegistry
from rag.v4.nodes.anchor_node import anchor_node
from rag.v4.nodes.eval_search_node import eval_search_node
from rag.v4.nodes.generator_node import generator_node
from rag.v4.nodes.merge_node import merge_node
from rag.v4.nodes.out_of_scope_node import out_of_scope_node
from rag.v4.nodes.retrieval_node import retrieval_node
from rag.v4.nodes.router_node import router_node
from rag.v4.nodes.schedule_filter_node import schedule_filter_node
from rag.v4.state import ConversationState, PipelineState


def _route_after_router(state: PipelineState) -> str:
    """Conditional edge: dispatch to the correct retrieval path."""
    function = state.get("function", "out_of_scope")
    if function == "out_of_scope":
        return "out_of_scope"
    elif function == "recurrent":
        return "anchor"
    else:
        # hybrid_course or semantic_general
        return "retrieval"


def _route_after_retrieval(state: PipelineState) -> str:
    """After retrieval, recurrent goes to schedule_filter; others go direct to generator."""
    function = state.get("function", "out_of_scope")
    if function == "recurrent":
        return "schedule_filter"
    return "generator"


def build_graph(registry: ComponentRegistry, tracer=None):
    """Build and compile the v4 pipeline graph.

    Args:
        registry: ComponentRegistry with all providers injected
        tracer: Optional V4Tracer (accepted for API compatibility; observability handled via state["trace"])

    Returns:
        Compiled LangGraph graph
    """
    graph = StateGraph(PipelineState)

    # Bind registry to every node via functools.partial
    def _bind(fn):
        return functools.partial(fn, registry=registry)

    # Add all nodes
    graph.add_node("router", _bind(router_node))
    graph.add_node("anchor", _bind(anchor_node))
    graph.add_node("eval_search", _bind(eval_search_node))
    graph.add_node("retrieval", _bind(retrieval_node))
    graph.add_node("schedule_filter", _bind(schedule_filter_node))
    graph.add_node("merge", _bind(merge_node))
    graph.add_node("generator", _bind(generator_node))
    graph.add_node("out_of_scope", _bind(out_of_scope_node))

    # Entry point
    graph.set_entry_point("router")

    # Conditional dispatch after router
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "out_of_scope": "out_of_scope",
            "anchor": "anchor",
            "retrieval": "retrieval",
        }
    )

    # Recurrent path before retrieval
    graph.add_edge("anchor", "eval_search")
    graph.add_edge("eval_search", "retrieval")

    # After retrieval: recurrent → schedule_filter, others → generator
    graph.add_conditional_edges(
        "retrieval",
        _route_after_retrieval,
        {
            "schedule_filter": "schedule_filter",
            "generator": "generator",
        }
    )

    # Recurrent post-retrieval path
    graph.add_edge("schedule_filter", "merge")
    graph.add_edge("merge", "generator")

    # Terminals
    graph.add_edge("generator", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile()


def build_graph_with_memory(registry: ComponentRegistry, checkpointer=None, tracer=None):
    """Build the v4 pipeline graph with conversation memory support (Phase 5).

    Node order:
    - router runs first (classifies function from the clean current query)
    - history_inject runs after router (enriches rewritten_query with context)
    - then the retrieval path runs with the enriched query
    - history_update runs at the END after generator/out_of_scope

    Args:
        registry: ComponentRegistry with all providers injected
        checkpointer: LangGraph checkpointer (MemorySaver, SqliteSaver, etc.)
        tracer: Optional V4Tracer (accepted for API compatibility; observability handled via state["trace"])

    Returns:
        Compiled LangGraph graph with checkpointing support
    """
    from rag.v4.nodes.history_inject_node import history_inject_node
    from rag.v4.nodes.history_update_node import history_update_node

    graph = StateGraph(ConversationState)

    def _bind(fn):
        return functools.partial(fn, registry=registry)

    # All pipeline nodes
    graph.add_node("router", _bind(router_node))
    graph.add_node("history_inject", _bind(history_inject_node))
    graph.add_node("anchor", _bind(anchor_node))
    graph.add_node("eval_search", _bind(eval_search_node))
    graph.add_node("retrieval", _bind(retrieval_node))
    graph.add_node("schedule_filter", _bind(schedule_filter_node))
    graph.add_node("merge", _bind(merge_node))
    graph.add_node("generator", _bind(generator_node))
    graph.add_node("out_of_scope", _bind(out_of_scope_node))
    graph.add_node("history_update", _bind(history_update_node))

    graph.set_entry_point("router")

    # Router → history_inject (always — then dispatch based on function)
    graph.add_edge("router", "history_inject")

    # history_inject → dispatch (same routing logic as build_graph, but from history_inject)
    graph.add_conditional_edges(
        "history_inject",
        _route_after_router,
        {
            "out_of_scope": "out_of_scope",
            "anchor": "anchor",
            "retrieval": "retrieval",
        }
    )

    # Recurrent path before retrieval
    graph.add_edge("anchor", "eval_search")
    graph.add_edge("eval_search", "retrieval")

    # After retrieval: recurrent → schedule_filter, others → generator
    graph.add_conditional_edges(
        "retrieval",
        _route_after_retrieval,
        {
            "schedule_filter": "schedule_filter",
            "generator": "generator",
        }
    )

    graph.add_edge("schedule_filter", "merge")
    graph.add_edge("merge", "generator")

    # After generator/out_of_scope: always update history then END
    graph.add_edge("generator", "history_update")
    graph.add_edge("out_of_scope", "history_update")
    graph.add_edge("history_update", END)

    kwargs = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer

    return graph.compile(**kwargs)
