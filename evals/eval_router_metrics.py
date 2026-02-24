"""Router evaluation metrics for TamuBot.

Computes three router quality metrics:
  - ECE  (Expected Calibration Error): measures how well category_confidence
         scores predict actual category extraction accuracy.
  - Intent F1: precision/recall/F1 for the semantic_intent=True binary flag.
  - Rewrite Cosine Gain: measures whether query rewriting improves retrieval
         alignment, using Voyage AI voyage-3 embeddings.

Can be run against:
  a) The existing 34-test dry-run eval suite (no MongoDB required)
  b) A golden set JSONL from generate_golden_set.py (--golden-set path)

Usage:
    # Run against built-in 34-test suite (dry-run, no MongoDB needed)
    python scripts/eval_router_metrics.py

    # Run against golden set (requires MongoDB + Voyage AI)
    python scripts/eval_router_metrics.py --golden-set tamu_data/logs/golden_set.jsonl

    # Skip rewrite gain (avoids Voyage AI API calls)
    python scripts/eval_router_metrics.py --skip-rewrite-gain
"""

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows (avoids cp1252 encode errors for → etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from db import router as router_mod


# ---------------------------------------------------------------------------
# ECE: Expected Calibration Error
# ---------------------------------------------------------------------------

def compute_ece(
    results: list[dict],
    n_bins: int = 10,
) -> dict:
    """Compute Expected Calibration Error for category_confidence scores.

    Bins results by category_confidence and measures how closely the mean
    confidence within each bin matches the actual accuracy in that bin.

    ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|

    Args:
        results: List of result dicts, each with:
                   category_confidence (float 0–1),
                   specific_categories_correct (bool)
        n_bins:  Number of equal-width bins (default 10).

    Returns:
        Dict with: ece, n_bins, bin_details (list of per-bin stats).
    """
    if not results:
        return {"ece": None, "n_bins": n_bins, "bin_details": []}

    bins: list[list[dict]] = [[] for _ in range(n_bins)]
    for r in results:
        conf = float(r.get("category_confidence", 0.0))
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append(r)

    ece = 0.0
    n_total = len(results)
    bin_details = []

    for i, b in enumerate(bins):
        if not b:
            bin_details.append({
                "bin": i,
                "range": [round(i / n_bins, 2), round((i + 1) / n_bins, 2)],
                "count": 0,
                "mean_conf": None,
                "accuracy": None,
                "gap": None,
            })
            continue

        mean_conf = sum(float(r.get("category_confidence", 0.0)) for r in b) / len(b)
        accuracy = sum(1 for r in b if r.get("specific_categories_correct", False)) / len(b)
        gap = abs(accuracy - mean_conf)

        ece += (len(b) / n_total) * gap
        bin_details.append({
            "bin": i,
            "range": [round(i / n_bins, 2), round((i + 1) / n_bins, 2)],
            "count": len(b),
            "mean_conf": round(mean_conf, 4),
            "accuracy": round(accuracy, 4),
            "gap": round(gap, 4),
        })

    return {
        "ece": round(ece, 4),
        "n_bins": n_bins,
        "interpretation": (
            "< 0.05 → well-calibrated, "
            "0.05–0.15 → moderate miscalibration, "
            "> 0.15 → poorly calibrated"
        ),
        "bin_details": bin_details,
    }


# ---------------------------------------------------------------------------
# Intent F1
# ---------------------------------------------------------------------------

def compute_intent_f1(results: list[dict]) -> dict:
    """Compute precision, recall, and F1 for the semantic_intent=True class.

    Treats semantic_intent extraction as binary classification:
      - Positive class: semantic_intent=True (advisory/subjective queries)
      - Uses expected_semantic_intent vs semantic_intent_actual

    Args:
        results: List of result dicts with:
                   semantic_intent_expected (bool),
                   semantic_intent_actual (bool)

    Returns:
        Dict with: precision, recall, f1, tp, fp, fn, tn, support.
    """
    tp = fp = fn = tn = 0
    for r in results:
        expected = bool(r.get("semantic_intent_expected", r.get("expected_semantic_intent", False)))
        actual = bool(r.get("semantic_intent_actual", False))

        if expected and actual:
            tp += 1
        elif not expected and actual:
            fp += 1
        elif expected and not actual:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "support": tp + fn,  # total actual positives
    }


# ---------------------------------------------------------------------------
# Rewrite Cosine Gain
# ---------------------------------------------------------------------------

def compute_rewrite_gain(
    results: list[dict],
    target_chunks: list[dict] | None = None,
) -> dict:
    """Measure whether query rewriting improves alignment with retrieved chunks.

    For each result that has a rewritten query, embeds both the original query
    and the rewritten query using Voyage AI voyage-3, then measures cosine
    similarity against the first retrieved chunk (as a proxy for target).

    If target_chunks is provided, uses those as targets instead.
    Requires VOYAGE_API_KEY to be set.

    Args:
        results:       List of result dicts with 'query' and 'rewritten_query'.
        target_chunks: Optional list of target chunk dicts (matched by index).

    Returns:
        Dict with: mean_gain, median_gain, pct_improved, n_pairs.
    """
    try:
        import voyageai
    except ImportError:
        return {"error": "voyageai package not installed", "mean_gain": None}

    if not config.VOYAGE_API_KEY:
        return {"error": "VOYAGE_API_KEY not set", "mean_gain": None}

    vo = voyageai.Client(api_key=config.VOYAGE_API_KEY)

    pairs_with_rewrite = [
        (r["query"], r["rewritten_query"], i)
        for i, r in enumerate(results)
        if r.get("rewritten_query") and r["rewritten_query"] != r.get("query", "")
    ]

    if not pairs_with_rewrite:
        return {"mean_gain": 0.0, "median_gain": 0.0, "pct_improved": 0.0, "n_pairs": 0}

    # Determine target texts
    target_texts: list[str] = []
    if target_chunks and len(target_chunks) >= len(pairs_with_rewrite):
        for _, _, idx in pairs_with_rewrite:
            chunk = target_chunks[idx] if idx < len(target_chunks) else None
            target_texts.append(chunk.get("content", "") if chunk else "")
    else:
        # Use rewritten query itself as a self-similarity proxy —
        # in absence of a ground-truth chunk, we measure query vs rewrite similarity
        target_texts = [orig for orig, _, _ in pairs_with_rewrite]

    original_queries = [orig for orig, _, _ in pairs_with_rewrite]
    rewritten_queries = [rew for _, rew, _ in pairs_with_rewrite]

    print(f"  Embedding {len(original_queries)} original queries...")
    orig_embeds = vo.embed(original_queries, model="voyage-3", input_type="query").embeddings

    print(f"  Embedding {len(rewritten_queries)} rewritten queries...")
    rew_embeds = vo.embed(rewritten_queries, model="voyage-3", input_type="query").embeddings

    print(f"  Embedding {len(target_texts)} target texts...")
    target_embeds = vo.embed(target_texts, model="voyage-3", input_type="document").embeddings

    gains = []
    for i in range(len(pairs_with_rewrite)):
        cos_orig = _cosine(orig_embeds[i], target_embeds[i])
        cos_rew = _cosine(rew_embeds[i], target_embeds[i])
        gains.append(cos_rew - cos_orig)

    mean_gain = sum(gains) / len(gains)
    sorted_gains = sorted(gains)
    n = len(sorted_gains)
    median_gain = (sorted_gains[n // 2] if n % 2 == 1
                   else (sorted_gains[n // 2 - 1] + sorted_gains[n // 2]) / 2)
    pct_improved = sum(1 for g in gains if g > 0) / len(gains)

    return {
        "mean_gain": round(mean_gain, 5),
        "median_gain": round(median_gain, 5),
        "pct_improved": round(pct_improved, 4),
        "n_pairs": len(gains),
    }


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Run router on test suite and collect results
# ---------------------------------------------------------------------------

def run_router_on_suite(
    suite: list[dict],
    verbose: bool = True,
) -> list[dict]:
    """Run the router on a list of test cases and return result dicts.

    Args:
        suite: List of dicts with at minimum 'query' and 'expected_function'.
               Optionally: expected_semantic_intent, expected_specific_categories.

    Returns:
        List of result dicts suitable for ECE, Intent F1, and Rewrite Gain.
    """
    results = []
    for i, tc in enumerate(suite, 1):
        query = tc.get("query", tc.get("question", ""))
        if verbose:
            print(f"  [{i:2d}/{len(suite)}] {query[:60]}...")
            sys.stdout.flush()

        t0 = time.perf_counter()
        try:
            rr = router_mod.classify_query(query)
            error = None
        except Exception as e:
            rr = router_mod.RouterResult(rewritten_query=query, category_confidence=0.0)
            error = str(e)
        latency_ms = (time.perf_counter() - t0) * 1000

        expected_cats = tc.get("expected_specific_categories", tc.get("specific_categories_expected", []))
        cats_correct = (
            set(expected_cats) <= set(rr.specific_categories)
            if expected_cats else True
        )

        result = {
            # Query
            "query": query,
            "rewritten_query": rr.rewritten_query,
            # Expected
            "function_expected": tc.get("function_expected", tc.get("expected_function", "")),
            "expected_semantic_intent": tc.get("expected_semantic_intent", False),
            "specific_categories_expected": expected_cats,
            # Actual
            "function_actual": rr.function,
            "semantic_intent_actual": rr.semantic_intent,
            "semantic_type_actual": rr.semantic_type,
            "specific_categories_actual": rr.specific_categories,
            "category_confidence": rr.category_confidence,
            "retrieval_mode_actual": rr.retrieval_mode,
            # Correctness
            "function_correct": rr.function == tc.get("function_expected", tc.get("expected_function", "")),
            "specific_categories_correct": cats_correct,
            # Timing
            "latency_router_ms": round(latency_ms, 1),
            "error": error,
        }
        results.append(result)

        if verbose:
            ok = "[OK]" if result["function_correct"] else "[FAIL]"
            print(f"       {ok} fn={rr.function} conf={rr.category_confidence:.2f}")

    return results


def _load_suite_from_eval_pipeline() -> list[dict]:
    """Load the 34-test suite from eval_pipeline.py as plain dicts."""
    from scripts.eval_pipeline import TEST_SUITE
    return [
        {
            "query": tc.query,
            "function_expected": tc.function_expected,
            "expected_semantic_intent": tc.expected_semantic_intent,
            "expected_specific_categories": tc.expected_specific_categories,
        }
        for tc in TEST_SUITE
    ]


def _load_golden_set(path: Path) -> list[dict]:
    """Load a golden set JSONL file."""
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    ece_result: dict,
    f1_result: dict,
    rewrite_result: dict,
    n_total: int,
    fn_accuracy: float,
) -> None:
    print("\n" + "=" * 60)
    print("  ROUTER EVAL METRICS REPORT")
    print("=" * 60)

    print(f"\nFunction Accuracy:   {fn_accuracy:.1%}  ({round(fn_accuracy * n_total)}/{n_total})")

    print(f"\nExpected Calibration Error (ECE): {ece_result.get('ece', 'N/A')}")
    print(f"  {ece_result.get('interpretation', '')}")
    non_empty_bins = [b for b in ece_result.get("bin_details", []) if b["count"] > 0]
    if non_empty_bins:
        print(f"  Non-empty bins:")
        for b in non_empty_bins:
            print(f"    [{b['range'][0]:.1f}–{b['range'][1]:.1f}] "
                  f"n={b['count']:2d}  conf={b['mean_conf']:.3f}  acc={b['accuracy']:.3f}  "
                  f"gap={b['gap']:.3f}")

    print(f"\nIntent F1 (semantic_intent=True class):")
    print(f"  Precision: {f1_result['precision']:.3f}")
    print(f"  Recall:    {f1_result['recall']:.3f}")
    print(f"  F1:        {f1_result['f1']:.3f}")
    print(f"  Support:   {f1_result['support']} positive cases")
    print(f"  TP={f1_result['tp']}  FP={f1_result['fp']}  FN={f1_result['fn']}  TN={f1_result['tn']}")

    if rewrite_result.get("error"):
        print(f"\nRewrite Cosine Gain: SKIPPED ({rewrite_result['error']})")
    elif rewrite_result.get("n_pairs", 0) == 0:
        print(f"\nRewrite Cosine Gain: N/A (no rewritten queries found)")
    else:
        print(f"\nRewrite Cosine Gain (n={rewrite_result['n_pairs']} pairs):")
        print(f"  Mean gain:     {rewrite_result['mean_gain']:+.5f}")
        print(f"  Median gain:   {rewrite_result['median_gain']:+.5f}")
        print(f"  Pct improved:  {rewrite_result['pct_improved']:.1%}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TamuBot router evaluation metrics")
    parser.add_argument("--golden-set", type=Path,
                        help="Path to golden_set.jsonl (default: use built-in 34-test suite)")
    parser.add_argument("--skip-rewrite-gain", action="store_true",
                        help="Skip Rewrite Cosine Gain (no Voyage AI calls)")
    parser.add_argument("--output", type=Path,
                        help="Write results JSON to this path")
    args = parser.parse_args()

    if args.golden_set:
        print(f"Loading golden set from {args.golden_set}...")
        suite = _load_golden_set(args.golden_set)
    else:
        print("Using built-in 34-test suite (eval_pipeline.py TEST_SUITE)...")
        suite = _load_suite_from_eval_pipeline()

    print(f"\nRunning router on {len(suite)} test cases...")
    results = run_router_on_suite(suite, verbose=True)

    fn_accuracy = sum(1 for r in results if r["function_correct"]) / len(results) if results else 0.0

    print("\nComputing ECE...")
    ece = compute_ece(results)

    print("Computing Intent F1...")
    f1 = compute_intent_f1(results)

    if args.skip_rewrite_gain:
        rewrite = {"error": "skipped via --skip-rewrite-gain", "mean_gain": None}
    else:
        print("Computing Rewrite Cosine Gain (requires Voyage AI)...")
        rewrite = compute_rewrite_gain(results)

    print_report(ece, f1, rewrite, n_total=len(results), fn_accuracy=fn_accuracy)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump({
                "n_total": len(results),
                "fn_accuracy": round(fn_accuracy, 4),
                "ece": ece,
                "intent_f1": f1,
                "rewrite_gain": rewrite,
                "results": results,
            }, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
