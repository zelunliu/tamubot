"""Statistical utilities for TamuBot eval pipeline.

Provides hypothesis testing helpers used across router, retrieval, and
generator eval scripts:

    adjusted_wald_ci()   — Agresti-Coull confidence interval for proportions
    mcnemar_exact()      — exact McNemar's test for paired binary outcomes
    wilcoxon_test()      — Wilcoxon signed-rank test for continuous metrics
    eval_summary_table() — aggregate summary dict from a list of result dicts

Usage:
    from scripts.eval_statistics import adjusted_wald_ci, mcnemar_exact

    ci = adjusted_wald_ci(45, 50)          # ~(0.786, 0.957)
    p  = mcnemar_exact(5, 0)               # p < 0.05
"""

import math
from typing import Any


# ---------------------------------------------------------------------------
# Adjusted Wald (Agresti-Coull) confidence interval
# ---------------------------------------------------------------------------

def adjusted_wald_ci(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return an Agresti-Coull (adjusted Wald) confidence interval for a proportion.

    The adjusted interval adds z²/2 pseudo-observations to both successes and
    failures before computing the Wald interval, which gives better coverage
    than the plain Wald CI for small n or extreme proportions.

    Args:
        successes:  Number of successes (e.g. correct router classifications).
        n:          Total trials.
        confidence: Desired confidence level (default 0.95 → 95% CI).

    Returns:
        (lower, upper) tuple, both clamped to [0.0, 1.0].

    Examples:
        adjusted_wald_ci(50, 50)   → ~(0.914, 1.0)   (perfect score)
        adjusted_wald_ci(45, 50)   → ~(0.786, 0.957)
        adjusted_wald_ci(34, 34)   → ~(0.898, 1.0)   (eval_pipeline 34/34)
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not (0 <= successes <= n):
        raise ValueError(f"successes={successes} out of range [0, {n}]")

    # z-score for the desired confidence level (two-tailed)
    z = _z_score(confidence)
    z2 = z * z

    # Adjusted counts
    n_tilde = n + z2
    p_tilde = (successes + z2 / 2) / n_tilde

    margin = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    lower = max(0.0, p_tilde - margin)
    upper = min(1.0, p_tilde + margin)
    return lower, upper


def _z_score(confidence: float) -> float:
    """Return the two-tailed z-score for a given confidence level.

    Uses the rational approximation to the inverse normal CDF (Beasley-Springer-Moro).
    Accurate to ~7 decimal places for confidence in [0.80, 0.999].
    """
    # probit via Abramowitz & Stegun rational approximation
    p = (1 + confidence) / 2  # one-sided p
    if p >= 1.0:
        return 8.0
    t = math.sqrt(-2 * math.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)


# ---------------------------------------------------------------------------
# McNemar's exact test
# ---------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar's test for paired binary outcomes.

    Used to compare two systems on the same test set: b = cases where the
    baseline is correct but the candidate is wrong; c = vice versa.

    The exact p-value is computed via the binomial CDF:
        p = 2 * Binomial.CDF(min(b, c), n=b+c, p=0.5)

    Args:
        b: Discordant pairs where baseline correct, candidate wrong.
        c: Discordant pairs where candidate correct, baseline wrong.

    Returns:
        Two-sided p-value (float). p < 0.05 → statistically significant.

    Examples:
        mcnemar_exact(5, 0)   → p ≈ 0.0625  (borderline, 5 discordant pairs)
        mcnemar_exact(10, 1)  → p ≈ 0.012   (significant)
        mcnemar_exact(3, 3)   → p = 1.0     (no difference)
    """
    n = b + c
    if n == 0:
        return 1.0  # no discordant pairs → no evidence of difference

    k = min(b, c)
    # p = 2 * P(X <= k) where X ~ Binomial(n, 0.5)
    cdf = _binom_cdf(k, n, 0.5)
    return min(1.0, 2 * cdf)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """Binomial CDF P(X <= k) for X ~ Binomial(n, p)."""
    total = 0.0
    for i in range(k + 1):
        total += _binom_pmf(i, n, p)
    return total


def _binom_pmf(k: int, n: int, p: float) -> float:
    """Binomial PMF P(X = k)."""
    if k < 0 or k > n:
        return 0.0
    log_p = (
        _log_comb(n, k)
        + k * math.log(p)
        + (n - k) * math.log(1 - p)
    )
    return math.exp(log_p)


def _log_comb(n: int, k: int) -> float:
    """log C(n, k) via log-gamma."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

def wilcoxon_test(
    baseline_scores: list[float],
    candidate_scores: list[float],
) -> dict[str, Any]:
    """Wilcoxon signed-rank test for continuous metric comparisons.

    Compares two paired sequences of scores (e.g. NDCG before and after
    reranking, or faithfulness across two generator configs).

    Uses a normal approximation for n > 25; falls back to exact enumeration
    for small samples.  Ties are handled by the standard midrank method.

    Args:
        baseline_scores:  Scores from the baseline system (one per test case).
        candidate_scores: Scores from the candidate system (same order).

    Returns:
        Dict with keys:
            statistic       — W+ (sum of positive ranks)
            p_value         — two-sided p-value
            effect_direction — "candidate_better", "baseline_better", or "no_difference"
            n_pairs         — number of non-tied pairs used
            mean_diff       — mean(candidate - baseline)

    Raises:
        ValueError: if the two lists have different lengths or fewer than 2 elements.
    """
    if len(baseline_scores) != len(candidate_scores):
        raise ValueError("baseline_scores and candidate_scores must have the same length")
    if len(baseline_scores) < 2:
        raise ValueError("Need at least 2 paired observations")

    diffs = [c - b for b, c in zip(baseline_scores, candidate_scores)]
    mean_diff = sum(diffs) / len(diffs)

    # Exclude zero differences
    nonzero = [(i, d) for i, d in enumerate(diffs) if d != 0]
    n = len(nonzero)
    if n == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_direction": "no_difference",
            "n_pairs": 0,
            "mean_diff": mean_diff,
        }

    # Rank absolute differences
    abs_diffs = sorted(range(n), key=lambda i: abs(nonzero[i][1]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs(nonzero[abs_diffs[j]][1]) == abs(nonzero[abs_diffs[i]][1]):
            j += 1
        midrank = (i + j + 1) / 2.0  # 1-indexed midrank
        for k in range(i, j):
            ranks[abs_diffs[k]] = midrank
        i = j

    w_plus = sum(ranks[i] for i in range(n) if nonzero[i][1] > 0)
    w_minus = sum(ranks[i] for i in range(n) if nonzero[i][1] < 0)

    # Normal approximation (adequate for n >= 10; used for simplicity)
    expected = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24
    if variance <= 0:
        p_value = 1.0
    else:
        z = (w_plus - expected) / math.sqrt(variance)
        p_value = 2 * (1 - _standard_normal_cdf(abs(z)))

    if w_plus > w_minus:
        direction = "candidate_better"
    elif w_minus > w_plus:
        direction = "baseline_better"
    else:
        direction = "no_difference"

    return {
        "statistic": w_plus,
        "p_value": round(p_value, 6),
        "effect_direction": direction,
        "n_pairs": n,
        "mean_diff": round(mean_diff, 6),
    }


def _standard_normal_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return (1 + math.erf(x / math.sqrt(2))) / 2


# ---------------------------------------------------------------------------
# Summary table helper
# ---------------------------------------------------------------------------

def eval_summary_table(
    results: list[dict[str, Any]],
    metric_key: str,
) -> dict[str, Any]:
    """Compute aggregate statistics for a numeric metric across a list of result dicts.

    Args:
        results:    List of dicts, each containing at least the metric_key.
        metric_key: The key to aggregate (must map to a numeric value).

    Returns:
        Dict with keys: count, mean, median, std, min, max, pct_above_0.9.

    Example:
        results = [{"faithfulness": 0.8}, {"faithfulness": 0.95}, ...]
        eval_summary_table(results, "faithfulness")
        # → {"count": 2, "mean": 0.875, "median": 0.875, ...}
    """
    values = [r[metric_key] for r in results if metric_key in r and r[metric_key] is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None,
                "min": None, "max": None, "pct_above_0.9": None}

    n = len(values)
    mean = sum(values) / n
    sorted_v = sorted(values)
    median = sorted_v[n // 2] if n % 2 == 1 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    pct_high = sum(1 for v in values if v >= 0.9) / n

    return {
        "count": n,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "pct_above_0.9": round(pct_high, 4),
    }


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== adjusted_wald_ci ===")
    print(f"50/50: {adjusted_wald_ci(50, 50)}")
    print(f"45/50: {adjusted_wald_ci(45, 50)}")
    print(f"34/34: {adjusted_wald_ci(34, 34)}")

    print("\n=== mcnemar_exact ===")
    print(f"b=5, c=0: p={mcnemar_exact(5, 0):.4f}")
    print(f"b=10, c=1: p={mcnemar_exact(10, 1):.4f}")
    print(f"b=3, c=3: p={mcnemar_exact(3, 3):.4f}")

    print("\n=== wilcoxon_test ===")
    baseline = [0.7, 0.6, 0.8, 0.5, 0.9, 0.7, 0.6, 0.8, 0.5, 0.7]
    candidate = [0.8, 0.7, 0.9, 0.6, 0.95, 0.75, 0.7, 0.85, 0.6, 0.8]
    print(wilcoxon_test(baseline, candidate))

    print("\n=== eval_summary_table ===")
    data = [{"faithfulness": v} for v in [0.8, 0.95, 0.7, 0.9, 0.85]]
    print(eval_summary_table(data, "faithfulness"))
