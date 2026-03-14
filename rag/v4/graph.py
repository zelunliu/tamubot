"""Build the v4 LangGraph state machine."""
from __future__ import annotations
import functools
from typing import Optional

from langgraph.graph import StateGraph, END

from rag.v4.state import PipelineState
from rag.v4.interfaces import ComponentRegistry
from rag.v4.nodes.router_node import router_node
from rag.v4.nodes.anchor_node import anchor_node
from rag.v4.nodes.eval_search_node import eval_search_node
from rag.v4.nodes.retrieval_node import retrieval_node
from rag.v4.nodes.schedule_filter_node import schedule_filter_node
from rag.v4.nodes.merge_node import merge_node
from rag.v4.nodes.generator_node import generator_node
from rag.v4.nodes.out_of_scope_node import out_of_scope_node


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
        tracer: Optional V4Tracer (ignored until Phase 4)

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
