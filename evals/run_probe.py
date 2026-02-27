"""Lightweight probe script — run 1-N queries through the full RAG pipeline with Langfuse tracing.

Useful for iterating on prompt changes, retrieval tuning, or spot-checking regressions
without running the full Streamlit app.

Usage:
    # Ad-hoc query
    python evals/run_probe.py --query "What is the grading breakdown for CSCE 638?"

    # Named tests from TEST_SUITE (1-based IDs)
    python evals/run_probe.py --test-ids 1 3 7

    # Run all named tests
    python evals/run_probe.py --suite all

    # Tag traces for before/after comparison
    python evals/run_probe.py --query "..." --tag "generator_v2"

    # Also run RAGAS faithfulness+relevancy scoring (async, ~30s)
    python evals/run_probe.py --test-ids 1 --ragas
"""

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repo root is on sys.path when invoked from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# UTF-8 stdout on Windows (avoids cp1252 encode errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from rag.router import route_retrieve_rerank
from rag.generator import generate
from rag.observability import get_langfuse, run_ragas_background
from evals.eval_pipeline import TEST_SUITE, TestCase  # noqa: F401  (TestCase re-exported for callers)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(query: str, max_len: int = 40) -> str:
    """Collapse non-alphanumeric runs to underscores, trim to max_len."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", query)
    return slug[:max_len].strip("_")


def _trace_url(trace_id: str) -> str:
    return f"{config.LANGFUSE_BASE_URL}/trace/{trace_id}"


# ---------------------------------------------------------------------------
# Core probe runner
# ---------------------------------------------------------------------------

def run_probe(
    query: str,
    tag: str | None = None,
    session_id: str | None = None,
    ragas: bool = False,
    index: int = 1,
    total: int = 1,
) -> dict:
    """Run one query through the full 3-stage pipeline and print a summary.

    Args:
        query:      The user query to probe.
        tag:        Optional trace tag for before/after comparisons.
        session_id: Groups all probes in a multi-query run under one session.
        ragas:      If True, kick off async RAGAS faithfulness+relevancy scoring.
        index:      1-based position in the current batch (for display).
        total:      Total queries in the batch (for display).

    Returns:
        Summary dict with function, chunk count, trace URL, etc.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_name = f"Probe_{ts}_{_sanitize(query)}"
    tags = ["probe"]
    if tag:
        tags.append(tag)

    # --- Langfuse trace ---
    lf = get_langfuse()
    trace = None
    trace_id = None
    if lf is not None:
        trace = lf.trace(
            name=trace_name,
            input={"query": query},
            metadata={
                "tags": tags,
                "session_id": session_id or f"probe_{ts}",
                "tag": tag,
            },
        )
        trace_id = trace.id
    else:
        print(
            "  [WARNING] Langfuse not configured (LANGFUSE_PUBLIC_KEY unset) "
            "— pipeline runs without tracing.",
            file=sys.stderr,
        )

    # --- Stage 1+2: route → retrieve → rerank ---
    t0 = time.time()
    reranked, router_result = route_retrieve_rerank(query, trace=trace)
    retrieval_elapsed = time.time() - t0

    # --- Stage 3: generate (blocking) ---
    t1 = time.time()
    answer = generate(
        results=reranked,
        question=query,
        function=router_result.function,
        course_ids=router_result.course_ids or None,
        semantic_type=router_result.semantic_type,
        trace=trace,
    )
    generation_elapsed = time.time() - t1

    # Attach final answer to trace
    if trace is not None:
        try:
            trace.update(output=answer[:500])
        except Exception:
            pass

    # --- Optional RAGAS background scoring ---
    if ragas and trace_id:
        contexts = [doc.get("content", "") for doc in reranked]
        run_ragas_background(query, contexts, answer, trace_id)

    # Flush so trace appears in Langfuse immediately
    if lf is not None:
        lf.flush()

    # --- Derive display values ---
    courses_str = ", ".join(router_result.course_ids) if router_result.course_ids else "(none)"
    cats_str = (
        ", ".join(router_result.specific_categories)
        if router_result.specific_categories
        else "(none)"
    )
    has_citations = "[Source" in answer
    # Gate 1 result: N/A for functions that don't require citations
    no_citation_fns = {"out_of_scope", "metadata_default"}
    if router_result.function in no_citation_fns:
        gate1_status = "N/A"
    else:
        gate1_status = "PASS" if has_citations else "MISSING"

    # --- Print summary ---
    print(f"\n[{index}/{total}] \"{query}\"")
    print(f"  → function: {router_result.function} | courses: {courses_str} | categories: {cats_str}")
    print(
        f"  → {len(reranked)} chunks after rerank "
        f"| retrieval: {retrieval_elapsed:.1f}s | generation: {generation_elapsed:.1f}s"
    )
    print(f"  → Gate1 citations: {gate1_status} | answer: {len(answer)} chars")
    if trace_id:
        print(f"  → Trace: {_trace_url(trace_id)}")
    if ragas and trace_id:
        print("  → RAGAS: scoring in background (~30s), check Langfuse for scores")

    return {
        "query": query,
        "function": router_result.function,
        "course_ids": router_result.course_ids,
        "specific_categories": router_result.specific_categories,
        "n_chunks": len(reranked),
        "answer_len": len(answer),
        "has_citations": has_citations,
        "gate1_status": gate1_status,
        "retrieval_elapsed": round(retrieval_elapsed, 2),
        "generation_elapsed": round(generation_elapsed, 2),
        "trace_id": trace_id,
        "trace_url": _trace_url(trace_id) if trace_id else None,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run probe queries through the full TamuBot RAG pipeline with Langfuse tracing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="Ad-hoc query string")
    group.add_argument(
        "--test-ids",
        type=int,
        nargs="+",
        metavar="N",
        help="1-based test case IDs from TEST_SUITE in eval_pipeline.py",
    )
    group.add_argument("--suite", choices=["all"], help="Run the full TEST_SUITE")

    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional Langfuse tag for before/after comparisons (e.g. 'generator_v2')",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Also run async RAGAS faithfulness+relevancy scoring (~30s, appears in Langfuse)",
    )
    args = parser.parse_args()

    # Build the list of queries to run
    if args.query:
        queries: list[str] = [args.query]
    elif args.suite == "all":
        queries = [tc.query for tc in TEST_SUITE]
    else:
        queries = []
        suite_size = len(TEST_SUITE)
        for tid in args.test_ids:
            if 1 <= tid <= suite_size:
                queries.append(TEST_SUITE[tid - 1].query)
            else:
                print(
                    f"[WARNING] Test ID {tid} out of range (valid: 1–{suite_size})",
                    file=sys.stderr,
                )

    if not queries:
        print("No queries to run.", file=sys.stderr)
        sys.exit(1)

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"probe_{session_ts}"
    total = len(queries)

    print(f"\nRunning {total} probe quer{'ies' if total != 1 else 'y'} (session: {session_id})")
    if args.tag:
        print(f"Tag: {args.tag}")
    if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        print("[WARNING] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — no tracing.")

    results: list[dict] = []
    for i, query in enumerate(queries, 1):
        try:
            result = run_probe(
                query=query,
                tag=args.tag,
                session_id=session_id,
                ragas=args.ragas,
                index=i,
                total=total,
            )
            results.append(result)
        except Exception as exc:
            print(f"\n[{i}/{total}] ERROR for query: {query!r}")
            print(f"  → {type(exc).__name__}: {exc}")

    print(f"\nDone. {len(results)}/{total} queries completed.")


if __name__ == "__main__":
    main()
