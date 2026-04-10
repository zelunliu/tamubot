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
import time
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
        "latency_ms", "recall_at_k", "f1_at_k", "context_precision",
    ]
    aggregates: dict = {}
    for m in metrics:
        values = [r[m] for r in results if r.get(m) is not None]
        if values:
            aggregates[f"avg_{m}"] = round(sum(values) / len(values), 4)
    aggregates["n_queries"] = len(results)
    return aggregates


# ---------------------------------------------------------------------------
# Retrieval — via the production eval graph (same code path as normal runs)
# ---------------------------------------------------------------------------

from rag.graph.pipeline import run_pipeline_eval  # noqa: E402


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
    span=None,
) -> Optional[dict]:
    """Run router + retrieval via the production eval graph. Returns result dict or None.

    RAGAS is intentionally NOT run here — it runs after span.end() in the loop
    so it doesn't inflate the Langfuse trace latency.

    Args:
        span: Open Langfuse observation — passed to run_pipeline_eval() so the
              graph's CallbackHandler nests router/retrieval spans under it.
    """
    if not question:
        return None

    if ragas_enabled and not reference:
        print(f"  [{i:2d}/{total}] SKIP (no reference_answer): {question[:60]}")
        return None

    print(f"  [{i:2d}/{total}] {question[:65]}...")

    try:
        t0 = time.perf_counter()
        chunks, router_result, timing_ms = run_pipeline_eval(question, trace=span)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:
        print(f"    Pipeline error: {e}")
        return None

    if not router_result.requires_retrieval:
        print(f"    Skip: {router_result.function} has no retrieval")
        return None

    # top_k is a metric evaluation cutoff — slice the ranked chunks from pipeline
    eval_chunks = chunks[:top_k] if top_k is not None else chunks

    emb = compute_embedding_metrics(question, eval_chunks, threshold)

    print(
        f"    prec={emb['precision_at_k']:.3f}  hit={emb['hit_rate_at_k']:.0f}"
        f"  tokens={emb['retrieved_tokens']}  latency={latency_ms:.0f}ms"
    )

    return {
        "query":            question,
        "_chunks":          eval_chunks,      # kept for RAGAS in loop, excluded from results
        "precision_at_k":   emb["precision_at_k"],
        "hit_rate_at_k":    emb["hit_rate_at_k"],
        "retrieved_tokens": emb["retrieved_tokens"],
        "latency_ms":       latency_ms,
        "recall_at_k":      None,             # filled by loop after RAGAS
        "f1_at_k":          None,
        "context_precision": None,
        "router_function":  router_result.function,
        "course_ids":       router_result.course_ids,
    }


def _score_trace(
    lf,
    trace_id: str,
    row: dict,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    top_k: Optional[int],
    threshold: float,
) -> None:
    """Post numeric metrics as scores on a Langfuse trace via create_score().

    Called after span.end() so scores don't block trace latency measurement.
    RAGAS scores (context_precision, context_recall) are posted by
    compute_retrieval_ragas() directly — not duplicated here.
    """
    # Per-query retrieval metrics (embedding-based, cheap)
    for name in ("precision_at_k", "hit_rate_at_k", "retrieved_tokens",
                 "recall_at_k", "f1_at_k"):
        value = row.get(name)
        if value is not None:
            lf.create_score(trace_id=trace_id, name=name, value=float(value))

    # Run-level config — same value for every item but visible as score columns
    if chunk_size is not None:
        lf.create_score(trace_id=trace_id, name="chunk_size", value=float(chunk_size))
    if chunk_overlap is not None:
        lf.create_score(trace_id=trace_id, name="chunk_overlap", value=float(chunk_overlap))
    lf.create_score(trace_id=trace_id, name="top_k",
                    value=float(top_k if top_k is not None else -1))
    lf.create_score(trace_id=trace_id, name="threshold", value=float(threshold))


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
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    description: Optional[str] = None,
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

        # Open Langfuse span BEFORE pipeline call.
        # The span's trace_id is passed to run_pipeline_eval() → CallbackHandler,
        # which nests router + retrieval node spans under this trace — same
        # structure as production runs.
        span = None
        if lf:
            try:
                span = lf.start_observation(
                    name=f"chunking_eval_{run_name}",
                    input=question,
                    metadata={"experiment": experiment, "run_name": run_name},
                )
            except Exception as e:
                logger.warning(f"Langfuse span open failed for '{question[:40]}': {e}")

        row = _run_one_query(
            question, reference, top_k, threshold, ragas_enabled,
            i, len(golden_items), span=span,
        )

        # End span BEFORE RAGAS — trace latency = router + retrieval only.
        trace_id = None
        if lf and span is not None:
            try:
                if row is not None:
                    span.update(
                        output={"router_function": row["router_function"],
                                "n_chunks": len(row["_chunks"])},
                        metadata={"experiment": experiment, "run_name": run_name,
                                  "router_function": row["router_function"],
                                  "course_ids": row["course_ids"]},
                    )
                span.end()
                trace_id = span.trace_id
            except Exception as e:
                logger.warning(f"Langfuse span end failed for '{question[:40]}': {e}")

        if row is None:
            continue

        # Run RAGAS after span.end() — doesn't inflate trace latency.
        # compute_retrieval_ragas() posts context_precision + context_recall
        # scores directly to Langfuse via trace_id.
        if ragas_enabled and row["_chunks"] and reference:
            contexts = [c.get("content", "") for c in row["_chunks"]]
            ragas_scores = compute_retrieval_ragas(
                question=question, contexts=contexts, reference=reference,
                trace_id=trace_id,
            )
            recall = ragas_scores.get("context_recall")
            precision_ragas = ragas_scores.get("context_precision")
            f1 = _compute_f1(row["precision_at_k"], recall) if recall is not None else None
            row["recall_at_k"] = recall
            row["f1_at_k"] = f1
            row["context_precision"] = precision_ragas
            recall_str = f"  recall={recall:.3f}" if recall is not None else ""
            print(f"    ...{recall_str}")

        # Strip internal _chunks before storing
        row.pop("_chunks", None)
        results.append(row)

        # Post embedding-based scores + config metadata to Langfuse
        if lf and trace_id:
            try:
                _score_trace(lf, trace_id, row, chunk_size, chunk_overlap, top_k, threshold)
                lf.api.dataset_run_items.create(
                    run_name=run_name,
                    run_description=description,
                    metadata={
                        "chunk_size":    chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "top_k":         top_k if top_k is not None else "auto",
                        "threshold":     threshold,
                    },
                    dataset_item_id=_item_id(question),
                    trace_id=trace_id,
                    observation_id=span.id,
                )
            except Exception as e:
                logger.warning(f"Langfuse scoring failed for '{question[:40]}': {e}")

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

    header = f"  {'Query':<42} {'Prec':>6} {'Hit':>5} {'Tokens':>7} {'Lat(ms)':>8}"
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
        lat = r.get("latency_ms")
        lat_str = f"{lat:>8.0f}" if lat is not None else f"{'N/A':>8}"
        print(
            f"  {r.get('query', '')[:42]:<42}"
            f" {r['precision_at_k']:>6.3f}"
            f" {r['hit_rate_at_k']:>5.0f}"
            f" {r['retrieved_tokens']:>7d}"
            f" {lat_str}"
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
        "--chunk-size", type=int, default=None,
        help="Chunk token size used during ingestion (stored in Langfuse run metadata)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=None,
        help="Chunk overlap tokens used during ingestion (stored in Langfuse run metadata)",
    )
    parser.add_argument(
        "--description", type=str, default=None,
        help="Human-readable goal or notes for this run (stored in Langfuse run description)",
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
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        description=args.description,
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
