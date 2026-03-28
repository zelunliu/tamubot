"""Verify mem0/cache integration: run the same query twice, check for cache hits.

Usage:
    PYTHONPATH=. python tools/verify_mem0_cache.py
"""
from __future__ import annotations
import time

import config

print(f"MEM0_ENABLED           = {config.MEM0_ENABLED}")
print(f"SESSION_CACHE_ENABLED  = {config.SESSION_CACHE_ENABLED}")
print(f"USE_TAMU_API           = {config.USE_TAMU_API}")
print()

from rag.v4.pipeline_v4 import run_pipeline_v4_with_memory
from rag.v4.session import SessionManager
import rag.v4.pipeline_v4 as pv4

SESSION_ID = "verify-cache-001"
session_mgr = SessionManager()
thread_config = session_mgr.get_thread_config(SESSION_ID)

QUERY = "What is the grading breakdown for CSCE 221?"
FOLLOW_UP = "What are the prerequisites for that course?"


def run_and_inspect(query: str, label: str) -> dict:
    t0 = time.perf_counter()
    run_pipeline_v4_with_memory(query, thread_config=thread_config)
    elapsed = (time.perf_counter() - t0) * 1000

    # Read node_trace from graph state
    graph = pv4._memory_graph
    state = {}
    if graph:
        snap = graph.get_state(thread_config)
        state = snap.values if snap and snap.values else {}

    node_trace = state.get("node_trace", [])
    timing_ms = state.get("timing_ms", {})

    print(f"{'─'*60}")
    print(f"{label}")
    print(f"  Query      : {query}")
    print(f"  Wall time  : {elapsed:.0f} ms")
    print(f"  node_trace : {node_trace}")
    router_ms   = timing_ms.get("router_node", 0)
    ret_ms      = timing_ms.get("retrieval_node", 0)
    print(f"  timing_ms  : router={router_ms:.0f}ms  retrieval={ret_ms:.0f}ms")

    hits = [n for n in node_trace if "cache_hit" in n]
    if hits:
        print(f"  ✅ Cache hits : {hits}")
    else:
        print(f"  ℹ  No cache hits (first run or cache disabled)")
    print()
    return state


def check_answer_cache(state: dict, query: str) -> None:
    from rag.v4.cache_utils import normalize_query
    answer_cache = state.get("answer_cache", {})
    key = normalize_query(query)
    if key in answer_cache:
        print(f"  ✅ answer_cache has entry for '{key[:40]}...'")
        print(f"     → {answer_cache[key][:120]}...")
    else:
        print(f"  ⚠  answer_cache has no entry for '{key[:40]}' (keys: {list(answer_cache.keys())[:3]})")


print("=" * 60)
print("Pass 1 — First query (cold, no cache)")
print("=" * 60)
state1 = run_and_inspect(QUERY, "Turn 1 (cold)")
check_answer_cache(state1, QUERY)

print()
print("=" * 60)
print("Pass 2 — Same query again (should hit router + retrieval cache)")
print("=" * 60)
state2 = run_and_inspect(QUERY, "Turn 2 (warm, should hit cache)")

print()
print("=" * 60)
print("Pass 3 — Follow-up query (tests mem0 context injection)")
print("=" * 60)
state3 = run_and_inspect(FOLLOW_UP, "Turn 3 (follow-up — check mem0 context)")
