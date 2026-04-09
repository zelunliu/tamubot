"""Retrieval-only chunking benchmark.

Measures precision_at_k, recall_at_k (RAGAS ContextRecall), f1_at_k,
hit_rate_at_k, and retrieved_tokens per query. Logs per-query traces and
a run-level aggregate trace to Langfuse.

Usage:
    python evals/eval_chunking.py \\
        --golden-set tamu_data/evals/golden_sets/golden_20260313_draft_v1.jsonl \\
        [--experiment chunk_600_ov100] \\
        [--top-k 7] \\
        [--threshold 0.85] \\
        [--ragas] \\
        [--output tamu_data/evals/reports/chunking_YYYYMMDD.json]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger("tamubot.eval_chunking")


# ---------------------------------------------------------------------------
# Golden set loader
# ---------------------------------------------------------------------------

def load_golden_set(path: Path) -> list[dict]:
    """Load a golden set JSONL file. Returns list of question dicts."""
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Embedding-based metrics (cheap, always computed)
# ---------------------------------------------------------------------------

def compute_embedding_metrics(
    query: str,
    chunks: list[dict],
    threshold: float = 0.85,
    _labels: Optional[list[bool]] = None,
) -> dict:
    """Compute embedding-based retrieval metrics without an LLM.

    Args:
        query:     User query string.
        chunks:    Reranked chunk dicts (must have 'content' key).
        threshold: Voyage-3 cosine similarity threshold (default 0.85).
        _labels:   Pre-computed relevance labels (skips Voyage AI; for tests).

    Returns:
        Dict with keys: precision_at_k (float), hit_rate_at_k (float),
        retrieved_tokens (int).
    """
    if not chunks:
        return {"precision_at_k": 0.0, "hit_rate_at_k": 0.0, "retrieved_tokens": 0}

    from evals.eval_retrieval_metrics import label_relevant

    labels = _labels if _labels is not None else label_relevant(query, chunks, threshold)
    k = len(labels)
    n_relevant = sum(labels)

    return {
        "precision_at_k": round(n_relevant / k, 4) if k > 0 else 0.0,
        "hit_rate_at_k": 1.0 if n_relevant > 0 else 0.0,
        "retrieved_tokens": sum(len(c.get("content", "")) // 4 for c in chunks),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return round(2.0 * precision * recall / (precision + recall), 4)


def _compute_aggregates(results: list[dict]) -> dict:
    """Mean of each numeric metric across all query results."""
    metrics = [
        "precision_at_k", "hit_rate_at_k", "retrieved_tokens",
        "recall_at_k", "f1_at_k", "context_precision",
    ]
    aggregates: dict = {}
    for m in metrics:
        values = [r[m] for r in results if r.get(m) is not None]
        if values:
            aggregates[f"avg_{m}"] = round(sum(values) / len(values), 4)
    aggregates["n_queries"] = len(results)
    return aggregates


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

from rag.router import classify_query, compute_dynamic_k  # noqa: E402
from rag.tools.mongo import hybrid_search, semantic_search  # noqa: E402
from rag.tools.voyage import rerank  # noqa: E402


def retrieve_for_query(
    query: str,
    rr,
    top_k: Optional[int] = None,
) -> list[dict]:
    """Run hybrid/semantic search + rerank for a single query.

    Args:
        query:  Original user query (used as fallback if rr.rewritten_query is empty).
        rr:     RouterResult from classify_query().
        top_k:  Override for rerank_k. If None, uses compute_dynamic_k().

    Returns:
        Reranked list of chunk dicts, or [] when the function has no retrieval
        path (e.g. out_of_scope, recurrent without course_ids).
    """
    dk = compute_dynamic_k(rr.function, len(rr.course_ids))
    retrieve_k = dk["retrieve_k"]
    rerank_k = top_k if top_k is not None else dk["rerank_k"]

    search_query = rr.rewritten_query or query

    if rr.function == "semantic_general":
        pre_results = semantic_search(search_query, retrieve_k)
    elif rr.course_ids:
        pre_results = hybrid_search(search_query, rr.course_ids[0], retrieve_k)
    else:
        return []

    return rerank(search_query, pre_results, rerank_k)


# ---------------------------------------------------------------------------
# Langfuse logging
# ---------------------------------------------------------------------------

from rag import get_langfuse  # noqa: E402
from rag.tools.langfuse import compute_retrieval_ragas  # noqa: E402


def _log_query_to_langfuse(lf, row: dict, run_name: str) -> None:
    """Create a Langfuse trace for one query and score all non-None metrics."""
    try:
        trace = lf.trace(
            name="retrieval_eval",
            input={"query": row["query"]},
            tags=["chunking_eval", run_name],
            session_id=run_name,
        )
        for metric in (
            "precision_at_k", "hit_rate_at_k", "retrieved_tokens",
            "recall_at_k", "f1_at_k", "context_precision", "context_recall",
        ):
            value = row.get(metric)
            if value is not None:
                lf.create_score(trace_id=trace.id, name=metric, value=float(value))
    except Exception as e:
        logger.warning(f"Langfuse per-query logging failed: {e}")


def _log_aggregates_to_langfuse(lf, results: list[dict], run_name: str) -> None:
    """Log run-level mean metrics as a summary trace in Langfuse."""
    try:
        aggregates = _compute_aggregates(results)
        trace = lf.trace(
            name="retrieval_eval_aggregate",
            input={"run_name": run_name, "n_queries": len(results)},
            output=aggregates,
            tags=["chunking_eval", run_name, "aggregate"],
            session_id=run_name,
        )
        for metric, value in aggregates.items():
            if isinstance(value, float):
                lf.create_score(trace_id=trace.id, name=metric, value=value)
        lf.flush()
    except Exception as e:
        logger.warning(f"Langfuse aggregate logging failed: {e}")


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(
    golden_items: list[dict],
    experiment: str,
    top_k: Optional[int],
    threshold: float,
    ragas_enabled: bool,
    lf,
) -> tuple[list[dict], str]:
    """Run retrieval eval over all golden items.

    Args:
        golden_items:   List of golden set question dicts.
        experiment:     Experiment name prefix for Langfuse run name.
        top_k:          Override for rerank_k (None → use compute_dynamic_k).
        threshold:      Voyage-3 cosine similarity threshold for relevance.
        ragas_enabled:  If True, run RAGAS ContextPrecision + ContextRecall.
        lf:             Langfuse client (or None to skip logging).

    Returns:
        Tuple of (results list, run_name string).
    """
    run_name = f"{experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results: list[dict] = []

    for i, item in enumerate(golden_items, 1):
        query = item.get("question", item.get("query", ""))
        reference = item.get("reference_answer", "")

        if not query:
            continue
        if ragas_enabled and not reference:
            print(f"  [{i:2d}] SKIP (no reference_answer): {query[:60]}")
            continue

        print(f"  [{i:2d}/{len(golden_items)}] {query[:60]}...")

        try:
            rr = classify_query(query)
        except Exception as e:
            print(f"    Router error: {e}")
            continue

        if not rr.requires_retrieval:
            print(f"    Skip: {rr.function} has no retrieval")
            continue

        try:
            chunks = retrieve_for_query(query, rr, top_k=top_k)
        except Exception as e:
            print(f"    Retrieval error: {e}")
            continue

        emb = compute_embedding_metrics(query, chunks, threshold)

        ragas_scores: dict = {}
        if ragas_enabled and chunks and reference:
            contexts = [c.get("content", "") for c in chunks]
            ragas_scores = compute_retrieval_ragas(
                question=query, contexts=contexts, reference=reference
            )

        recall = ragas_scores.get("context_recall")
        precision = emb["precision_at_k"]
        f1 = _compute_f1(precision, recall) if recall is not None else None

        row = {
            "query": query,
            **emb,
            "recall_at_k": recall,
            "f1_at_k": f1,
            "context_precision": ragas_scores.get("context_precision"),
            "context_recall": recall,
        }
        results.append(row)

        if lf:
            _log_query_to_langfuse(lf, row, run_name)

        recall_str = f"  recall={recall:.3f}" if recall is not None else ""
        print(
            f"    prec={precision:.3f}  hit={emb['hit_rate_at_k']:.0f}"
            f"  tokens={emb['retrieved_tokens']}{recall_str}"
        )

    if lf and results:
        _log_aggregates_to_langfuse(lf, results, run_name)

    return results, run_name


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], run_name: str, aggregates: dict) -> None:
    """Print aligned per-query and aggregate metrics to stdout."""
    has_ragas = any(r.get("recall_at_k") is not None for r in results)
    print(f"\n{'='*80}")
    print(f"  RETRIEVAL EVAL: {run_name}  |  {len(results)} queries")
    print(f"{'='*80}")

    header = f"  {'Query':<42} {'Prec':>6} {'Hit':>5} {'Tokens':>7}"
    if has_ragas:
        header += f" {'Recall':>7} {'F1':>6}"
    print(header)
    print(f"  {'-'*78}")

    for r in results:
        recall_str = ""
        if has_ragas:
            rec = r.get("recall_at_k")
            f1 = r.get("f1_at_k")
            recall_str = (
                f" {rec:>7.3f} {f1:>6.3f}"
                if rec is not None and f1 is not None
                else f" {'N/A':>7} {'N/A':>6}"
            )
        print(
            f"  {r['query'][:42]:<42}"
            f" {r['precision_at_k']:>6.3f}"
            f" {r['hit_rate_at_k']:>5.0f}"
            f" {r['retrieved_tokens']:>7d}"
            f"{recall_str}"
        )

    print(f"{'='*80}")
    print("  AGGREGATES:")
    for k, v in aggregates.items():
        print(f"    {k:<30} {v}")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval-only chunking benchmark — compare chunking strategies by retrieval quality"
    )
    parser.add_argument(
        "--golden-set", type=Path, required=True,
        help="Path to golden set JSONL (e.g. tamu_data/evals/golden_sets/golden_*.jsonl)",
    )
    parser.add_argument(
        "--experiment", default="chunking_eval",
        help="Experiment name prefix for Langfuse run (default: chunking_eval)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Override rerank_k; default uses compute_dynamic_k() per query",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Voyage-3 cosine similarity threshold for relevance labels (default: 0.85)",
    )
    parser.add_argument(
        "--ragas", action="store_true",
        help="Enable RAGAS ContextPrecision + ContextRecall (costs LLM tokens, ~30s/query)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Write JSON results to this path (optional)",
    )
    args = parser.parse_args()

    if not args.golden_set.exists():
        print(f"ERROR: Golden set not found: {args.golden_set}")
        sys.exit(1)

    print(f"\nLoading golden set: {args.golden_set}")
    golden_items = load_golden_set(args.golden_set)
    print(f"  {len(golden_items)} items loaded")

    lf = get_langfuse()
    print(f"  Langfuse: {'connected' if lf else 'not configured (logging skipped)'}")

    print(
        f"\nRunning eval: ragas={'yes' if args.ragas else 'no'}"
        f"  threshold={args.threshold}"
        f"  top_k={args.top_k or 'auto'}\n"
    )

    results, run_name = run_eval(
        golden_items=golden_items,
        experiment=args.experiment,
        top_k=args.top_k,
        threshold=args.threshold,
        ragas_enabled=args.ragas,
        lf=lf,
    )

    if not results:
        print("No results produced. Check golden set and pipeline connectivity.")
        sys.exit(1)

    aggregates = _compute_aggregates(results)
    print_summary(results, run_name, aggregates)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(
                {"run_name": run_name, "aggregates": aggregates, "items": results},
                f, indent=2,
            )
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
