"""Retrieval-only chunking benchmark.

Measures precision_at_k, recall_at_k (RAGAS ContextRecall), f1_at_k,
hit_rate_at_k, and retrieved_tokens per query. Logs to a Langfuse dataset
experiment run — each query is one row, each metric is a column.

Usage:
    python evals/eval_chunking.py \\
        --golden-set tamu_data/evals/golden_sets/golden_20260313_draft_v1.jsonl \\
        [--experiment chunk_600_ov100] \\
        [--dataset chunking_golden_v1] \\
        [--top-k 7] \\
        [--threshold 0.35] \\
        [--ragas] \\
        [--output tamu_data/evals/reports/chunking_YYYYMMDD.json]
"""

import argparse
import hashlib
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
    threshold: float = 0.35,
    _labels: Optional[list[bool]] = None,
) -> dict:
    """Compute embedding-based retrieval metrics without an LLM.

    Args:
        query:     User query string.
        chunks:    Reranked chunk dicts (must have 'content' key).
        threshold: Voyage-3 cosine similarity threshold (default 0.35).
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
# Langfuse dataset upsert
# ---------------------------------------------------------------------------

from rag import get_langfuse  # noqa: E402
from rag.tools.langfuse import compute_retrieval_ragas  # noqa: E402


def _item_id(question: str) -> str:
    """Stable 16-char ID for a question — used for idempotent upsert."""
    return hashlib.md5(question.encode()).hexdigest()[:16]


def upsert_langfuse_dataset(lf, golden_items: list[dict], dataset_name: str):
    """Create-or-upsert a Langfuse dataset and upload all golden items.

    Dataset column layout (flat, Langfuse-friendly):
      input           — question string (renders as readable text column)
      expected_output — reference answer string (renders as readable text column)
      metadata        — flat dict of structured params (each key = a column)

    Returns the DatasetClient, or None on failure.
    """
    try:
        lf.create_dataset(
            name=dataset_name,
            description=(
                "TamuBot chunking eval — retrieval quality benchmark. "
                "Each item is a golden question with router expectations and reference answer."
            ),
        )
    except Exception:
        pass  # dataset already exists — that's fine

    uploaded = 0
    for item in golden_items:
        question = item.get("question", item.get("query", ""))
        if not question:
            continue
        try:
            lf.create_dataset_item(
                id=_item_id(question),
                dataset_name=dataset_name,
                # Flat strings → render as clean text columns in Langfuse UI
                input=question,
                expected_output=item.get("reference_answer") or "",
                # Structured params → each key becomes a searchable metadata column
                # (keep values short — Langfuse OTel propagation limit is 200 chars)
                metadata={
                    "expected_function": item.get("expected_function"),
                    "source_course_id":  item.get("source_course_id"),
                    "source_category":   item.get("source_category"),
                    "stratum":           item.get("stratum"),
                    "category":          item.get("category"),
                },
            )
            uploaded += 1
        except Exception as e:
            logger.warning(f"Dataset item upsert failed for '{question[:40]}': {e}")

    print(f"  Langfuse dataset '{dataset_name}': {uploaded}/{len(golden_items)} items upserted.")

    try:
        return lf.get_dataset(dataset_name)
    except Exception as e:
        logger.warning(f"Could not fetch dataset after upsert: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-query runner (shared by Langfuse and fallback paths)
# ---------------------------------------------------------------------------

def _run_one_query(
    question: str,
    reference: str,
    top_k: Optional[int],
    threshold: float,
    ragas_enabled: bool,
    i: int,
    total: int,
) -> Optional[dict]:
    """Run retrieval + metrics for one question. Returns result dict or None to skip."""
    if not question:
        return None

    if ragas_enabled and not reference:
        print(f"  [{i:2d}/{total}] SKIP (no reference_answer): {question[:60]}")
        return None

    print(f"  [{i:2d}/{total}] {question[:65]}...")

    try:
        rr = classify_query(question)
    except Exception as e:
        print(f"    Router error: {e}")
        return None

    if not rr.requires_retrieval:
        print(f"    Skip: {rr.function} has no retrieval")
        return None

    try:
        chunks = retrieve_for_query(question, rr, top_k=top_k)
    except Exception as e:
        print(f"    Retrieval error: {e}")
        return None

    emb = compute_embedding_metrics(question, chunks, threshold)

    ragas_scores: dict = {}
    if ragas_enabled and chunks and reference:
        contexts = [c.get("content", "") for c in chunks]
        ragas_scores = compute_retrieval_ragas(
            question=question, contexts=contexts, reference=reference
        )

    recall = ragas_scores.get("context_recall")
    precision = emb["precision_at_k"]
    f1 = _compute_f1(precision, recall) if recall is not None else None

    recall_str = f"  recall={recall:.3f}" if recall is not None else ""
    print(
        f"    prec={precision:.3f}  hit={emb['hit_rate_at_k']:.0f}"
        f"  tokens={emb['retrieved_tokens']}{recall_str}"
    )

    return {
        "query":             question,
        "precision_at_k":    emb["precision_at_k"],
        "hit_rate_at_k":     emb["hit_rate_at_k"],
        "retrieved_tokens":  emb["retrieved_tokens"],
        "recall_at_k":       recall,
        "f1_at_k":           f1,
        "context_precision": ragas_scores.get("context_precision"),
        "router_function":   rr.function,
        "course_ids":        rr.course_ids,
    }


def _score_trace(span, row: dict) -> None:
    """Attach all numeric metrics as scores on a Langfuse trace.

    Each metric becomes a separate named score — in the Traces list the user
    can enable them as columns (Columns toggle → score names).
    """
    for name in ("precision_at_k", "hit_rate_at_k", "retrieved_tokens",
                 "recall_at_k", "f1_at_k", "context_precision"):
        value = row.get(name)
        if value is not None:
            span.score_trace(name=name, value=float(value))


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(
    golden_items: list[dict],
    experiment: str,
    dataset_name: str,
    top_k: Optional[int],
    threshold: float,
    ragas_enabled: bool,
    lf,
) -> tuple[list[dict], str]:
    """Run retrieval eval over all golden items.

    Langfuse layout when lf is set:
      • One trace per query in the Traces list — add score names as columns
        via the Columns toggle to see precision_at_k / hit_rate_at_k / etc.
      • Dataset items (input=question, expected_output=reference) linked to
        their traces via source_trace_id.
      • All traces tagged [experiment, run_name, "chunking_eval"] for filtering.

    Returns:
        Tuple of (results list, run_name string).
    """
    run_name = f"{experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if lf:
        upsert_langfuse_dataset(lf, golden_items, dataset_name)
        print(f"\nRunning eval: experiment={experiment!r}  run={run_name!r}\n")

    results: list[dict] = []

    for i, item in enumerate(golden_items, 1):
        question = item.get("question", item.get("query", ""))
        reference = item.get("reference_answer", "") or ""

        row = _run_one_query(question, reference, top_k, threshold, ragas_enabled, i, len(golden_items))
        if row is None:
            continue

        results.append(row)

        if lf:
            try:
                span = lf.start_observation(
                    name="retrieval_eval",
                    input=question,
                    output={k: v for k, v in row.items() if k != "query"},
                    metadata={
                        "experiment":      experiment,
                        "run_name":        run_name,
                        "router_function": row["router_function"],
                        "course_ids":      row["course_ids"],
                    },
                )
                _score_trace(span, row)
                span.end()
                # Link trace to its dataset item so the item shows the source trace
                lf.create_dataset_item(
                    id=_item_id(question),
                    dataset_name=dataset_name,
                    input=question,
                    expected_output=reference,
                    metadata={
                        "expected_function": item.get("expected_function"),
                        "source_course_id":  item.get("source_course_id"),
                        "source_category":   item.get("source_category"),
                        "stratum":           item.get("stratum"),
                        "category":          item.get("category"),
                    },
                    source_trace_id=span.trace_id,
                )
            except Exception as e:
                logger.warning(f"Langfuse logging failed for '{question[:40]}': {e}")

    if lf:
        lf.flush()

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
            f"  {r.get('query', '')[:42]:<42}"
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
        help="Experiment name in Langfuse (default: chunking_eval)",
    )
    parser.add_argument(
        "--dataset",
        help="Langfuse dataset name to upsert items into (default: stem of --golden-set filename)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Override rerank_k; default uses compute_dynamic_k() per query",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.35,
        help="Voyage-3 cosine similarity threshold for relevance labels (default: 0.35)",
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

    dataset_name = args.dataset or args.golden_set.stem

    print(f"\nLoading golden set: {args.golden_set}")
    golden_items = load_golden_set(args.golden_set)
    print(f"  {len(golden_items)} items loaded")

    lf = get_langfuse()
    print(f"  Langfuse: {'connected' if lf else 'not configured (logging skipped)'}")
    print(
        f"  Dataset:  {dataset_name}"
        f"  |  experiment: {args.experiment}"
        f"  |  threshold: {args.threshold}"
        f"  |  top_k: {args.top_k or 'auto'}"
        f"  |  ragas: {'yes' if args.ragas else 'no'}"
    )

    results, run_name = run_eval(
        golden_items=golden_items,
        experiment=args.experiment,
        dataset_name=dataset_name,
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
