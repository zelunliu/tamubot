"""Tiered generator evaluation for TamuBot.

Three-tier evaluation cascade for generator output quality:

  Tier 1 — Deterministic: regex citation check (fast, no API calls)
  Tier 2 — Embedding proxy: Voyage-3 cosine similarity vs reference answer
  Tier 3 — LLM-as-judge: RAGAS Faithfulness + AnswerRelevancy (only when Tier 2 is ambiguous)

Escalation logic:
  Tier 2 score >= 0.9  → PASS  (skip Tier 3)
  Tier 2 score <  0.6  → FAIL  (skip Tier 3)
  Tier 2 score in [0.6, 0.9)  → ambiguous → run Tier 3

Usage:
    # Evaluate a single answer
    python scripts/eval_generator_tiered.py \
        --question "What is the grading for CSCE 638?" \
        --answer "The grading is 40% homework, 30% midterm, 30% final." \
        --reference "CSCE 638 grading: 40% assignments, 30% midterm, 30% final exam." \
        --n-sources 3

    # Batch evaluation over golden set (requires MongoDB + full pipeline)
    python scripts/eval_generator_tiered.py --golden-set tamu_data/logs/golden_set.jsonl
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


# ---------------------------------------------------------------------------
# Tier 1: Deterministic citation check
# ---------------------------------------------------------------------------

def tier1_citation_check(answer: str, n_sources: int) -> dict[str, Any]:
    """Check that citations in the answer are well-formed and within valid range.

    Validates:
      - At least one [Source N] reference is present (if n_sources > 0)
      - All cited source IDs are in the range [1, n_sources]
      - No hallucinated or out-of-range source IDs

    Args:
        answer:    The generator's answer string.
        n_sources: Number of sources that were passed to the generator.

    Returns:
        Dict with: pass (bool), citation_count, valid_ids, invalid_ids, details.
    """
    if n_sources == 0:
        # No sources were provided — citations are neither expected nor valid
        citations_found = re.findall(r"\[Source\s+(\d+)\]", answer, re.IGNORECASE)
        return {
            "pass": len(citations_found) == 0,
            "citation_count": len(citations_found),
            "valid_ids": [],
            "invalid_ids": [int(c) for c in citations_found],
            "details": "No sources provided — citations should be absent.",
        }

    citations = re.findall(r"\[Source\s+(\d+)\]", answer, re.IGNORECASE)
    citation_ids = [int(c) for c in citations]

    valid_ids = [cid for cid in citation_ids if 1 <= cid <= n_sources]
    invalid_ids = [cid for cid in citation_ids if not (1 <= cid <= n_sources)]

    has_citations = len(citation_ids) > 0
    no_invalid = len(invalid_ids) == 0
    passed = has_citations and no_invalid

    return {
        "pass": passed,
        "citation_count": len(citation_ids),
        "valid_ids": sorted(set(valid_ids)),
        "invalid_ids": sorted(set(invalid_ids)),
        "details": (
            f"Found {len(citation_ids)} citation(s). "
            f"Valid: {valid_ids}. Invalid: {invalid_ids}."
            if citation_ids else
            f"No [Source N] citations found (n_sources={n_sources})."
        ),
    }


# ---------------------------------------------------------------------------
# Tier 2: Embedding cosine proxy
# ---------------------------------------------------------------------------

def tier2_embedding_similarity(answer: str, reference: str) -> float:
    """Compute Voyage-3 cosine similarity between answer and reference.

    Uses document embeddings for both texts (symmetric comparison).
    Returns a float in [0.0, 1.0]:
      >= 0.9  → strong semantic match → PASS
      <  0.6  → poor match → FAIL
      [0.6, 0.9) → ambiguous → escalate to Tier 3

    Args:
        answer:    Generator output.
        reference: Ground-truth reference answer.

    Returns:
        Cosine similarity float.
    """
    if not config.VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY not set — cannot run Tier 2 embedding similarity.")

    import voyageai
    vo = voyageai.Client(api_key=config.VOYAGE_API_KEY)

    embeds = vo.embed([answer, reference], model="voyage-3", input_type="document").embeddings
    return _cosine(embeds[0], embeds[1])


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Tier 3: LLM-as-judge (RAGAS)
# ---------------------------------------------------------------------------

def tier3_llm_judge(
    question: str,
    contexts: list[str],
    answer: str,
) -> dict[str, Any]:
    """Run RAGAS Faithfulness + AnswerRelevancy as LLM-as-judge.

    Only called when Tier 2 is ambiguous (score in [0.6, 0.9)).
    Reuses compute_ragas_metrics() from db/observability.py.

    Args:
        question: The original user query.
        contexts: Retrieved chunk texts passed to the generator.
        answer:   The generator's answer string.

    Returns:
        Dict from compute_ragas_metrics(): {"faithfulness": float, "answer_relevancy": float, ...}
        or {"error": str} on failure.
    """
    try:
        from rag import compute_ragas_metrics
        scores = compute_ragas_metrics(question=question, contexts=contexts, answer=answer)
        return scores if scores else {"error": "RAGAS returned empty scores"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# Tier 2 thresholds
_T2_PASS_THRESHOLD = 0.9
_T2_FAIL_THRESHOLD = 0.6


def evaluate_answer(
    answer: str,
    reference: str,
    question: str,
    contexts: list[str],
    n_sources: int,
) -> dict[str, Any]:
    """Full 3-tier generator evaluation.

    Tier 1: Citation regex check (always runs)
    Tier 2: Embedding cosine similarity (runs if Voyage AI available)
    Tier 3: RAGAS LLM-as-judge (only when Tier 2 score is ambiguous)

    Args:
        answer:    Generator output text.
        reference: Ground-truth reference answer.
        question:  Original user query.
        contexts:  List of retrieved chunk texts passed to the generator.
        n_sources: Number of source chunks available (for citation validation).

    Returns:
        Dict with:
          tier1:         citation check result dict
          tier2_score:   cosine similarity float or None
          tier2_verdict: "PASS", "FAIL", or "AMBIGUOUS"
          tier3:         RAGAS scores dict or None (None = not needed)
          overall:       "PASS", "FAIL", or "NEEDS_REVIEW"
          tiers_run:     list of int (which tiers were executed)
    """
    result: dict[str, Any] = {
        "question": question[:200],
        "answer_preview": answer[:200],
        "n_sources": n_sources,
        "tier1": None,
        "tier2_score": None,
        "tier2_verdict": None,
        "tier3": None,
        "overall": None,
        "tiers_run": [],
    }

    # ── Tier 1 ─────────────────────────────────────────────────────────
    t1 = tier1_citation_check(answer, n_sources)
    result["tier1"] = t1
    result["tiers_run"].append(1)

    # Citation failure is a hard FAIL — skip higher tiers
    if not t1["pass"]:
        result["overall"] = "FAIL"
        result["tier2_verdict"] = "skipped"
        return result

    # ── Tier 2 ─────────────────────────────────────────────────────────
    if not reference:
        # No reference available — skip Tier 2, go straight to Tier 3
        result["tier2_verdict"] = "skipped_no_reference"
    else:
        try:
            sim = tier2_embedding_similarity(answer, reference)
            result["tier2_score"] = round(sim, 4)
            result["tiers_run"].append(2)

            if sim >= _T2_PASS_THRESHOLD:
                result["tier2_verdict"] = "PASS"
                result["overall"] = "PASS"
                return result
            elif sim < _T2_FAIL_THRESHOLD:
                result["tier2_verdict"] = "FAIL"
                result["overall"] = "FAIL"
                return result
            else:
                result["tier2_verdict"] = "AMBIGUOUS"
        except RuntimeError as e:
            # Voyage AI not available
            result["tier2_verdict"] = f"skipped: {e}"

    # ── Tier 3 ─────────────────────────────────────────────────────────
    if contexts:
        t3 = tier3_llm_judge(question, contexts, answer)
        result["tier3"] = t3
        result["tiers_run"].append(3)

        if "error" in t3:
            result["overall"] = "NEEDS_REVIEW"
        else:
            # Use faithfulness as the primary quality signal
            faithfulness = t3.get("faithfulness", 0.0)
            relevancy = t3.get("answer_relevancy", 0.0)
            if faithfulness >= 0.7 and relevancy >= 0.7:
                result["overall"] = "PASS"
            elif faithfulness < 0.4 or relevancy < 0.4:
                result["overall"] = "FAIL"
            else:
                result["overall"] = "NEEDS_REVIEW"
    else:
        result["overall"] = "NEEDS_REVIEW"
        result["tier3"] = {"error": "no contexts provided for Tier 3 judge"}

    return result


# ---------------------------------------------------------------------------
# Batch evaluation over golden set
# ---------------------------------------------------------------------------

def evaluate_golden_set(
    golden_set: list[dict],
    run_full_pipeline: bool = True,
) -> dict[str, Any]:
    """Evaluate generator quality over a full golden set.

    For each golden question, runs the full pipeline (router → retrieval → generator),
    then evaluates with the 3-tier cascade.

    Args:
        golden_set:        List of golden question dicts (from generate_golden_set.py).
        run_full_pipeline: If True, run retrieval + generation (requires MongoDB).
                           If False, only run Tier 1 on existing 'answer' field.

    Returns:
        Aggregate metrics dict.
    """
    from rag import route_retrieve_rerank, generate

    item_results = []
    for i, item in enumerate(golden_set, 1):
        question = item.get("question", item.get("query", ""))
        reference = item.get("reference_answer", "")
        if not question:
            continue

        print(f"  [{i:2d}/{len(golden_set)}] {question[:60]}...")
        sys.stdout.flush()

        if run_full_pipeline:
            try:
                chunks, rr = route_retrieve_rerank(question)
                answer = generate(
                    chunks, question,
                    function=rr.function,
                    course_ids=rr.course_ids,
                    semantic_type=rr.semantic_type,
                )
                contexts = [c.get("content", "") for c in chunks]
                n_sources = len(chunks)
            except Exception as e:
                print(f"    Pipeline error: {e}")
                item_results.append({"question": question, "error": str(e)})
                continue
        else:
            answer = item.get("answer", "")
            contexts = [item.get("context", "")]
            n_sources = item.get("n_sources", 1)

        eval_result = evaluate_answer(
            answer=answer,
            reference=reference,
            question=question,
            contexts=contexts,
            n_sources=n_sources,
        )
        eval_result["stratum"] = item.get("stratum", "unknown")
        eval_result["expected_function"] = item.get("expected_function", "")
        item_results.append(eval_result)

        print(f"    overall={eval_result['overall']}  "
              f"tiers_run={eval_result['tiers_run']}  "
              f"t2={eval_result.get('tier2_score', 'N/A')}")

    # Aggregates
    valid = [r for r in item_results if "overall" in r]
    if not valid:
        return {"error": "no valid results", "items": item_results}

    outcomes = [r["overall"] for r in valid]
    n_pass = outcomes.count("PASS")
    n_fail = outcomes.count("FAIL")
    n_review = outcomes.count("NEEDS_REVIEW")

    t1_pass = sum(1 for r in valid if r.get("tier1", {}).get("pass", False))
    t2_scores = [r["tier2_score"] for r in valid if r.get("tier2_score") is not None]
    t3_faith = [
        r["tier3"]["faithfulness"]
        for r in valid
        if r.get("tier3") and "faithfulness" in (r["tier3"] or {})
    ]

    return {
        "n_total": len(valid),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_needs_review": n_review,
        "pass_rate": round(n_pass / len(valid), 4) if valid else 0.0,
        "tier1_pass_rate": round(t1_pass / len(valid), 4) if valid else 0.0,
        "tier2_mean_cosine": round(sum(t2_scores) / len(t2_scores), 4) if t2_scores else None,
        "tier3_mean_faithfulness": round(sum(t3_faith) / len(t3_faith), 4) if t3_faith else None,
        "items": item_results,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TamuBot tiered generator evaluation")
    parser.add_argument("--question", help="Single question to evaluate")
    parser.add_argument("--answer", help="Answer text (used with --question)")
    parser.add_argument("--reference", default="", help="Reference answer (used with --question)")
    parser.add_argument("--context", nargs="*", default=[], help="Context strings (used with --question)")
    parser.add_argument("--n-sources", type=int, default=1,
                        help="Number of sources in the answer (default: 1)")
    parser.add_argument("--golden-set", type=Path,
                        help="Golden set JSONL for batch evaluation")
    parser.add_argument("--no-pipeline", action="store_true",
                        help="With --golden-set: skip pipeline, use 'answer' field from JSONL")
    parser.add_argument("--output", type=Path, help="Write results JSON to this path")
    args = parser.parse_args()

    if args.question:
        if not args.answer:
            print("ERROR: --answer is required with --question")
            sys.exit(1)

        print(f"Evaluating answer for: '{args.question[:60]}...'")
        result = evaluate_answer(
            answer=args.answer,
            reference=args.reference,
            question=args.question,
            contexts=args.context,
            n_sources=args.n_sources,
        )

        print(f"\n{'=' * 50}")
        print(f"  GENERATOR EVAL RESULT")
        print(f"{'=' * 50}")
        print(f"  Tier 1 (citation):   {'PASS' if result['tier1']['pass'] else 'FAIL'}")
        print(f"    {result['tier1']['details']}")
        if result["tier2_score"] is not None:
            print(f"  Tier 2 (embedding):  {result['tier2_score']:.4f} → {result['tier2_verdict']}")
        else:
            print(f"  Tier 2 (embedding):  {result['tier2_verdict']}")
        if result["tier3"]:
            print(f"  Tier 3 (RAGAS):      {result['tier3']}")
        print(f"\n  OVERALL: {result['overall']}")
        print(f"  Tiers run: {result['tiers_run']}")

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"\nResult written to {args.output}")

    elif args.golden_set:
        print(f"Loading golden set from {args.golden_set}...")
        golden = []
        with args.golden_set.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    golden.append(json.loads(line))
        print(f"Evaluating {len(golden)} golden questions...")
        summary = evaluate_golden_set(golden, run_full_pipeline=not args.no_pipeline)

        print(f"\n{'=' * 50}")
        print(f"  GENERATOR EVAL SUMMARY")
        print(f"{'=' * 50}")
        print(f"  Total:             {summary.get('n_total', 0)}")
        print(f"  Pass:              {summary.get('n_pass', 0)} ({summary.get('pass_rate', 0):.1%})")
        print(f"  Fail:              {summary.get('n_fail', 0)}")
        print(f"  Needs review:      {summary.get('n_needs_review', 0)}")
        print(f"  Tier 1 pass rate:  {summary.get('tier1_pass_rate', 'N/A')}")
        if summary.get("tier2_mean_cosine") is not None:
            print(f"  Tier 2 mean cos:   {summary['tier2_mean_cosine']:.4f}")
        if summary.get("tier3_mean_faithfulness") is not None:
            print(f"  Tier 3 faithfulness: {summary['tier3_mean_faithfulness']:.4f}")

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"\nResults written to {args.output}")

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
