"""Golden set label adjudication for TamuBot evaluation.

The golden set ground truth is derived mechanically from strata in generate_golden_set.py.
When the synthesis LLM produces a question that escapes its stratum framing (e.g. asks
about prerequisites under the metadata_default stratum), the label is wrong even though
the router is right.

This script:
  1. Loads the existing golden set JSONL
  2. Runs the router on each question (or reads from a prior router_metrics.json)
  3. For each case where the stratum label and router output disagree, asks Gemini to
     adjudicate which function label is semantically correct
  4. Writes a corrected golden set JSONL with updated expected_function,
     expected_specific_categories, and expected_semantic_intent

Usage:
    # Adjudicate from scratch (re-runs router, ~50 LLM calls for routing + adjudication)
    python scripts/adjudicate_golden_set.py \\
        --golden-set tamu_data/logs/golden_set.jsonl \\
        --output    tamu_data/logs/golden_set_v2.jsonl

    # Use cached router results (faster — skips re-routing)
    python scripts/adjudicate_golden_set.py \\
        --golden-set    tamu_data/logs/golden_set.jsonl \\
        --router-results tamu_data/logs/router_metrics.json \\
        --output         tamu_data/logs/golden_set_v2.jsonl

    # Dry-run: show what would change without writing
    python scripts/adjudicate_golden_set.py \\
        --golden-set tamu_data/logs/golden_set.jsonl \\
        --dry-run
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from google.genai import types

# ---------------------------------------------------------------------------
# Derivation matrix description (injected into the adjudicator prompt)
# ---------------------------------------------------------------------------

MATRIX_DESCRIPTION = """\
Function derivation matrix (pure Python rules, no LLM):
  course_ids  recurrent_search  semantic_intent  specific_categories  specific_only  → function
  empty       any               True             any                  any            → semantic_general
  empty       any               False            any                  any            → out_of_scope
  present     True              any              empty                —              → recurrent_default
  present     True              any              populated            True           → recurrent_specific
  present     True              any              populated            False          → recurrent_combined
  present     False             any              empty                —              → metadata_default
  present     False             any              populated            True           → metadata_specific
  present     False             any              populated            False          → metadata_combined

Key variables:
  course_ids        — course(s) the student is asking about ([] if none named)
  recurrent_search  — True ONLY when user wants to discover unknown courses using a named course as anchor
                      ("What course should I take with CS 638?")
  semantic_intent   — True if question is evaluative, advisory, opinion-based,
                      or a TAMU discovery query with no course_id
  specific_categories — syllabus categories explicitly targeted ([] for general overview)
  specific_only     — True if the question asks ONLY about those categories;
                      False if broad overview with a category as emphasis

Retrieval mode (separate from function):
  metadata  — exact index lookup, no vector search (all metadata_* functions)
  hybrid    — two-stage: anchor metadata fetch + corpus-wide hybrid (recurrent_* functions)
  semantic  — full-corpus vector search (semantic_general, no course_ids)
"""

# ---------------------------------------------------------------------------
# Adjudicator prompt
# ---------------------------------------------------------------------------

ADJUDICATOR_PROMPT = """\
You are a router label adjudicator for a Texas A&M University course assistant eval suite.

The eval suite assigns an "expected_function" label to each student question based on which
question-generation stratum it came from. Sometimes the synthesized question escapes its
stratum framing and the stratum label is semantically wrong — even though the router's
actual output is correct.

Your job: given a student question and two candidate function labels (the stratum label
and the router's actual output), decide which label is semantically correct for this question.
You may also propose a third label if both are wrong.

{matrix}

---

Student question: {question}

Stratum label (golden set "expected"): {stratum_label}
  Stratum extracted variables assumed: {stratum_vars}

Router actual output: {router_label}
  Router extracted variables: {router_vars}

---

Think step by step:
1. What is the student actually asking? (factual lookup, general overview, evaluative/opinion,
   or a TAMU discovery question with no specific course?)
2. Does the question name a specific course ID?
3. Does the question ask about a specific syllabus category, or is it a general overview request?
4. Is there any evaluative, advisory, or opinion language?
5. Which function label fits the derivation matrix for this question?

Output a JSON object with these fields:
{{
  "correct_label": "<one of the 8 function names>",
  "correct_specific_categories": ["CAT1", ...] or [],
  "correct_semantic_intent": true or false,
  "correct_specific_only": true or false,
  "reasoning": "<1-2 sentences explaining why>"
}}

Valid category names: COURSE_OVERVIEW, INSTRUCTOR, PREREQUISITES, LEARNING_OUTCOMES,
MATERIALS, GRADING, SCHEDULE, ATTENDANCE_AND_MAKEUP, AI_POLICY, UNIVERSITY_POLICIES,
SUPPORT_SERVICES

Respond with ONLY valid JSON, no other text.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stratum_vars_description(item: dict) -> str:
    """Human-readable description of what the stratum label implies."""
    fn = item.get("expected_function", "")
    cats = item.get("expected_specific_categories", [])
    si = item.get("expected_semantic_intent", False)
    return (
        f"course_ids={item.get('expected_course_ids', [])}, "
        f"semantic_intent={si}, specific_categories={cats}, "
        f"specific_only={'True' if '_specific' in fn else 'False' if '_combined' in fn else 'N/A'}"
    )


def _router_vars_description(r: dict) -> str:
    """Human-readable description of the router's actual extraction."""
    return (
        f"course_ids={r.get('function_actual_course_ids', r.get('course_ids_extracted', []))}, "
        f"semantic_intent={r.get('semantic_intent_actual', '?')}, "
        f"specific_categories={r.get('specific_categories_actual', [])}, "
        f"conf={r.get('category_confidence', r.get('category_confidence_actual', '?'))}"
    )


def adjudicate_case(
    question: str,
    stratum_label: str,
    router_label: str,
    stratum_vars: str,
    router_vars: str,
) -> dict | None:
    """Ask Gemini to adjudicate between the stratum label and router label."""
    client = config.get_genai_client()
    prompt = ADJUDICATOR_PROMPT.format(
        matrix=MATRIX_DESCRIPTION,
        question=question,
        stratum_label=stratum_label,
        stratum_vars=stratum_vars,
        router_label=router_label,
        router_vars=router_vars,
    )
    try:
        response = client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                max_output_tokens=512,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"  [WARN] Adjudication failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Router execution (used when no cached results are provided)
# ---------------------------------------------------------------------------

def run_router_on_golden_set(items: list[dict]) -> list[dict]:
    """Run the router on each golden set question and return per-case result dicts."""
    from rag import classify_query
    results = []
    for i, item in enumerate(items, 1):
        q = item.get("question", item.get("query", ""))
        print(f"  [{i:2d}/{len(items)}] {q[:70]}...")
        sys.stdout.flush()
        try:
            rr = classify_query(q)
            results.append({
                "query": q,
                "function_actual": rr.function,
                "semantic_intent_actual": rr.semantic_intent,
                "specific_categories_actual": rr.specific_categories,
                "category_confidence": rr.category_confidence,
                "function_correct": rr.function == item.get("expected_function", ""),
            })
        except Exception as e:
            results.append({
                "query": q,
                "function_actual": "error",
                "semantic_intent_actual": False,
                "specific_categories_actual": [],
                "category_confidence": 0.0,
                "function_correct": False,
                "error": str(e),
            })
        time.sleep(0.1)  # gentle rate limiting
    return results


# ---------------------------------------------------------------------------
# Main adjudication pipeline
# ---------------------------------------------------------------------------

def adjudicate(
    golden_path: Path,
    router_results_path: Path | None,
    output_path: Path | None,
    dry_run: bool,
    only_disagreements: bool,
) -> None:
    # Load golden set
    print(f"\nLoading golden set from {golden_path}...")
    with golden_path.open(encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    print(f"  {len(items)} items loaded.")

    # Load or compute router results
    if router_results_path and router_results_path.exists():
        print(f"\nLoading router results from {router_results_path}...")
        with router_results_path.open(encoding="utf-8") as f:
            data = json.load(f)
        router_results = data.get("results", [])
        print(f"  {len(router_results)} results loaded.")
    else:
        print(f"\nRunning router on {len(items)} questions...")
        router_results = run_router_on_golden_set(items)

    if len(router_results) != len(items):
        print(f"[ERROR] Mismatch: {len(items)} golden items vs {len(router_results)} router results.")
        sys.exit(1)

    # Identify disagreements
    disagreements = [
        i for i, (item, rr) in enumerate(zip(items, router_results))
        if item.get("expected_function", "") != rr.get("function_actual", "")
    ]
    agreements = [i for i in range(len(items)) if i not in disagreements]

    print(f"\n{'=' * 60}")
    print(f"  Disagreements: {len(disagreements)}/{len(items)}")
    print(f"  Agreements:    {len(agreements)}/{len(items)}")
    print(f"{'=' * 60}")

    if dry_run:
        print("\n  [DRY-RUN] Cases that would be adjudicated:")
        for i in disagreements:
            item = items[i]
            rr = router_results[i]
            q = item.get("question", "")
            print(f"  [{i+1:2d}] {q[:70]}")
            print(f"       stratum={item.get('expected_function')}  router={rr.get('function_actual')}")
        print("\n  [DRY-RUN] No output written.")
        return

    # Adjudicate each disagreement
    corrected_items = list(items)  # copy
    n_changed = 0
    n_kept_stratum = 0
    n_kept_router = 0
    n_third_option = 0

    print(f"\nAdjudicating {len(disagreements)} disagreements...")
    for idx in disagreements:
        item = items[idx]
        rr = router_results[idx]
        q = item.get("question", item.get("query", ""))
        stratum_label = item.get("expected_function", "")
        router_label = rr.get("function_actual", "")

        print(f"\n  [{idx+1:2d}] {q[:70]}")
        print(f"       stratum={stratum_label}  router={router_label}")

        verdict = adjudicate_case(
            question=q,
            stratum_label=stratum_label,
            router_label=router_label,
            stratum_vars=_stratum_vars_description(item),
            router_vars=_router_vars_description(rr),
        )

        if verdict is None:
            print(f"       -> SKIP (adjudication failed — keeping stratum label)")
            n_kept_stratum += 1
            continue

        correct_label = verdict.get("correct_label", stratum_label)
        reasoning = verdict.get("reasoning", "")
        print(f"       -> {correct_label}  [{reasoning[:80]}]")

        # Apply correction to the item
        corrected = dict(item)
        corrected["expected_function"] = correct_label
        corrected["expected_specific_categories"] = verdict.get("correct_specific_categories", [])
        corrected["expected_semantic_intent"] = verdict.get("correct_semantic_intent", item.get("expected_semantic_intent"))
        corrected["adjudicated"] = True
        corrected["adjudication_reasoning"] = reasoning
        corrected["original_expected_function"] = stratum_label
        corrected["router_actual_function"] = router_label
        corrected_items[idx] = corrected

        if correct_label == stratum_label:
            n_kept_stratum += 1
        elif correct_label == router_label:
            n_kept_router += 1
            n_changed += 1
        else:
            n_third_option += 1
            n_changed += 1

        time.sleep(0.1)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Adjudication summary ({len(disagreements)} cases):")
    print(f"    Kept stratum label:  {n_kept_stratum}")
    print(f"    Adopted router label: {n_kept_router}")
    print(f"    Third option chosen: {n_third_option}")
    print(f"    Total corrections:   {n_changed}")
    print(f"{'=' * 60}")

    # Write output
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for item in corrected_items:
                f.write(json.dumps(item) + "\n")
        print(f"\nCorrected golden set written to: {output_path}")

        # Also write a summary of changes
        changes_path = output_path.with_suffix(".changes.jsonl")
        with changes_path.open("w", encoding="utf-8") as f:
            for idx in disagreements:
                orig = items[idx]
                corr = corrected_items[idx]
                if orig.get("expected_function") != corr.get("expected_function"):
                    f.write(json.dumps({
                        "index": idx + 1,
                        "question": orig.get("question", ""),
                        "original_label": orig.get("expected_function"),
                        "corrected_label": corr.get("expected_function"),
                        "router_actual": router_results[idx].get("function_actual"),
                        "reasoning": corr.get("adjudication_reasoning", ""),
                    }) + "\n")
        print(f"Change log written to:          {changes_path}")
    else:
        print("\n[No --output specified — results not saved]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Adjudicate golden set labels using Gemini"
    )
    parser.add_argument("--golden-set", required=True, metavar="PATH",
                        help="Path to golden_set.jsonl")
    parser.add_argument("--router-results", metavar="PATH",
                        help="Path to router_metrics.json (cached router output). "
                             "If omitted, the router is re-run on all questions.")
    parser.add_argument("--output", metavar="PATH",
                        help="Output path for corrected JSONL "
                             "(default: <golden-set-stem>_v2.jsonl in same dir)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be adjudicated without writing output")
    parser.add_argument("--all", action="store_true",
                        help="Adjudicate ALL cases (not just disagreements). "
                             "Useful for a full re-label of the golden set.")
    args = parser.parse_args()

    golden_path = Path(args.golden_set)
    if not golden_path.exists():
        print(f"[ERROR] Golden set not found: {golden_path}")
        sys.exit(1)

    router_results_path = Path(args.router_results) if args.router_results else None

    if args.output:
        output_path = Path(args.output)
    elif not args.dry_run:
        output_path = golden_path.parent / (golden_path.stem + "_v2.jsonl")
    else:
        output_path = None

    adjudicate(
        golden_path=golden_path,
        router_results_path=router_results_path,
        output_path=output_path,
        dry_run=args.dry_run,
        only_disagreements=not args.all,
    )


if __name__ == "__main__":
    main()
