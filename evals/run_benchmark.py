"""End-to-end benchmark runner → versioned Excel + Markdown reports.

Runs a golden set JSONL through the full RAG pipeline, captures per-stage metrics,
and exports versioned reports for A/B experiment comparison.

Usage:
    python evals/run_benchmark.py \
        --golden-set tamu_data/evals/golden_sets/golden_20260311_v1.jsonl \
        --experiment-name cs600_ov100 \
        [--ragas]

Output:
    tamu_data/evals/reports/benchmark_{experiment_name}_{YYYYMMDD}.xlsx   (3 tabs)
    tamu_data/evals/reports/benchmark_{experiment_name}_{YYYYMMDD}.md
"""

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from rag import (
    FUNCTION_CATEGORY_STRATEGIES,
    RouterResult,
    classify_query,
    compute_dynamic_k,
    deduplicate_chunks,
    generate,
)

REPORTS_DIR = Path("tamu_data/evals/reports")


# ---------------------------------------------------------------------------
# Per-question result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRow:
    # Identity
    question: str
    stratum: str
    category: str
    source_crn: str

    # Ground truth
    expected_function: str

    # Router
    router_function: str
    router_function_correct: bool
    router_ms: float

    # Retrieval
    retrieved_correct: Optional[bool]  # None when source_crn is blank
    retrieval_ms: float

    # Generator
    answer_preview: str
    citation_pass: bool
    generator_ms: float
    total_ms: float

    # RAGAS (optional)
    ragas_faithfulness: Optional[float] = None
    ragas_relevancy: Optional[float] = None

    # Error
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _do_retrieval(rr: RouterResult, query: str) -> list[dict]:
    """Retrieval stage — mirrors eval_pipeline._do_retrieval."""
    from rag import reranker, search

    search_query = rr.rewritten_query or query

    if not rr.requires_retrieval:
        return []

    fn = rr.function
    dk = compute_dynamic_k(fn, len(rr.course_ids))
    retrieve_k = dk["retrieve_k"]
    rerank_k = dk["rerank_k"]
    course_ids = rr.course_ids

    if fn == "semantic_general":
        results = search.search_semantic(search_query, top_k=retrieve_k)
        reranked = reranker.rerank(search_query, results, top_k=rerank_k)
        return deduplicate_chunks(reranked)

    if fn == "out_of_scope" or not course_ids:
        return []

    strategy = FUNCTION_CATEGORY_STRATEGIES.get(fn)
    categories = strategy(rr) if strategy else list(config.DEFAULT_SUMMARY_CATEGORIES)

    if fn.startswith("recurrent_"):
        from rag.generator import generate_eval_search_string
        from rag.search import fetch_anchor_chunks
        anchor_chunks, _, _ = fetch_anchor_chunks(course_ids, categories)
        eval_query = generate_eval_search_string(
            anchor_chunks, search_query, rr.intent_type or "GENERAL"
        )
        all_results = search.hybrid_search(eval_query, filters=None, k=retrieve_k)
        anchor_ids = set(course_ids)
        discovery = [c for c in all_results if c.get("course_id") not in anchor_ids]
        reranked = reranker.rerank(eval_query, discovery, top_k=rerank_k)
        return deduplicate_chunks(anchor_chunks + reranked)

    if len(course_ids) == 1:
        return deduplicate_chunks(search.search_by_course_categories(course_ids[0], categories))

    combined: list[dict] = []
    for cid in course_ids:
        combined.extend(search.search_by_course_categories(cid, categories))
    return deduplicate_chunks(combined)


def run_one(item: dict, do_ragas: bool) -> BenchmarkRow:
    """Run one golden set item through router → retrieval → generation."""
    query = item["question"]
    source_crn = str(item.get("source_crn") or "")
    source_category = str(item.get("source_category") or "")
    expected_fn = str(item.get("expected_function", ""))

    t0 = time.perf_counter()
    error: Optional[str] = None
    chunks: list[dict] = []
    answer = ""

    # Router
    t_router = time.perf_counter()
    try:
        rr = classify_query(query)
    except Exception as e:
        rr = RouterResult(rewritten_query=query, category_confidence=0.0)
        error = f"router: {e}"
    router_ms = (time.perf_counter() - t_router) * 1000

    # Retrieval
    t_ret = time.perf_counter()
    if not error and rr.requires_retrieval:
        try:
            chunks = _do_retrieval(rr, query)
        except Exception as e:
            error = f"retrieval: {e}"
    retrieval_ms = (time.perf_counter() - t_ret) * 1000

    # Generation
    t_gen = time.perf_counter()
    if not error:
        try:
            if rr.function == "out_of_scope":
                answer = generate([], query, function="out_of_scope")
            elif chunks or rr.requires_retrieval:
                answer = generate(
                    chunks, query,
                    function=rr.function,
                    course_ids=rr.course_ids,
                    intent_type=rr.intent_type,
                )
        except Exception as e:
            error = f"generation: {e}"
    generator_ms = (time.perf_counter() - t_gen) * 1000
    total_ms = (time.perf_counter() - t0) * 1000

    # Recall — did source chunk surface in results?
    retrieved_correct: Optional[bool] = None
    if source_crn and not error:
        retrieved_correct = any(
            c.get("crn") == source_crn and c.get("category") == source_category
            for c in chunks
        )

    # RAGAS (optional)
    ragas_faithfulness: Optional[float] = None
    ragas_relevancy: Optional[float] = None
    if do_ragas and chunks and answer and not error:
        try:
            from rag import compute_ragas_metrics
            contexts = [c.get("content", "") for c in chunks if c.get("content")]
            scores = compute_ragas_metrics(question=query, contexts=contexts, answer=answer)
            ragas_faithfulness = scores.get("faithfulness")
            ragas_relevancy = scores.get("answer_relevancy")
        except Exception:
            pass  # non-fatal

    return BenchmarkRow(
        question=query,
        stratum=item.get("stratum", ""),
        category=item.get("category") or "",
        source_crn=source_crn,
        expected_function=expected_fn,
        router_function=rr.function,
        router_function_correct=(rr.function == expected_fn),
        router_ms=round(router_ms, 1),
        retrieved_correct=retrieved_correct,
        retrieval_ms=round(retrieval_ms, 1),
        answer_preview=(answer[:300] + "..." if len(answer) > 300 else answer),
        citation_pass=bool(re.search(r"\[Source \d+\]", answer)),
        generator_ms=round(generator_ms, 1),
        total_ms=round(total_ms, 1),
        ragas_faithfulness=ragas_faithfulness,
        ragas_relevancy=ragas_relevancy,
        error=error,
    )


# ---------------------------------------------------------------------------
# Output: Excel (3 tabs) + Markdown
# ---------------------------------------------------------------------------

def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _avg(rows: list[BenchmarkRow], attr: str) -> Optional[float]:
    vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def write_excel(
    rows: list[BenchmarkRow],
    experiment_name: str,
    golden_set_path: str,
    output_path: Path,
    do_ragas: bool,
) -> None:
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        print("ERROR: openpyxl required. Run: pip install openpyxl")
        sys.exit(1)

    wb = openpyxl.Workbook()
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    wrap = Alignment(wrap_text=True, vertical="top")

    n = len(rows)
    n_fn_correct = sum(1 for r in rows if r.router_function_correct)
    recall_cases = [r for r in rows if r.retrieved_correct is not None]
    recall_hits = sum(1 for r in recall_cases if r.retrieved_correct)
    citation_cases = [r for r in rows if r.router_function != "out_of_scope"]
    citation_pass = sum(1 for r in citation_cases if r.citation_pass)
    ragas_cases = [r for r in rows if r.ragas_faithfulness is not None]
    avg_faith = sum(r.ragas_faithfulness for r in ragas_cases) / len(ragas_cases) if ragas_cases else None

    # ── Tab 1: Summary ───────────────────────────────────────────────────
    ws_sum = wb.active
    ws_sum.title = "Summary"

    summary_rows = [
        ("Experiment", experiment_name),
        ("Date", datetime.now().strftime("%Y-%m-%d")),
        ("n_questions", n),
        ("Router accuracy",
         f"{n_fn_correct/n:.1%} ({n_fn_correct}/{n})" if n else "N/A"),
        ("Recall@k (crn+category)",
         f"{recall_hits/len(recall_cases):.1%} ({recall_hits}/{len(recall_cases)})" if recall_cases else "N/A"),
        ("Citation pass rate",
         f"{citation_pass/len(citation_cases):.1%} ({citation_pass}/{len(citation_cases)})" if citation_cases else "N/A"),
        ("Mean RAGAS faithfulness",
         f"{avg_faith:.2f}" if avg_faith is not None else "not run (use --ragas)"),
        ("Mean total latency (ms)", _avg(rows, "total_ms")),
        ("Mean router latency (ms)", _avg(rows, "router_ms")),
        ("Mean retrieval latency (ms)", _avg(rows, "retrieval_ms")),
        ("Mean generator latency (ms)", _avg(rows, "generator_ms")),
        ("Errors", sum(1 for r in rows if r.error)),
    ]

    for col, label in [(1, "Metric"), (2, "Value")]:
        cell = ws_sum.cell(row=1, column=col, value=label)
        cell.font = header_font
        cell.fill = header_fill
    ws_sum.column_dimensions["A"].width = 35
    ws_sum.column_dimensions["B"].width = 30

    for i, (metric, val) in enumerate(summary_rows, start=2):
        ws_sum.cell(row=i, column=1, value=metric)
        ws_sum.cell(row=i, column=2, value=val)

    # ── Tab 2: Per-Query ─────────────────────────────────────────────────
    ws_pq = wb.create_sheet("Per-Query")

    pq_cols = [
        ("question", 50),
        ("stratum", 25),
        ("category", 20),
        ("source_crn", 12),
        ("expected_function", 22),
        ("router_function", 22),
        ("router_function_correct", 12),
        ("retrieved_correct", 12),
        ("citation_pass", 12),
        ("ragas_faithfulness", 14),
        ("ragas_relevancy", 14),
        ("router_ms", 10),
        ("retrieval_ms", 12),
        ("generator_ms", 12),
        ("total_ms", 10),
        ("answer_preview", 60),
        ("error", 30),
        # human_judgment column: blank, user fills after viewing answers
        ("human_judgment", 14),
    ]

    for col_idx, (h, width) in enumerate(pq_cols, start=1):
        cell = ws_pq.cell(row=1, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        ws_pq.column_dimensions[cell.column_letter].width = width

    for row_idx, r in enumerate(rows, start=2):
        values = [
            r.question, r.stratum, r.category, r.source_crn,
            r.expected_function, r.router_function, r.router_function_correct,
            r.retrieved_correct, r.citation_pass,
            r.ragas_faithfulness, r.ragas_relevancy,
            r.router_ms, r.retrieval_ms, r.generator_ms, r.total_ms,
            r.answer_preview, r.error,
            "",  # human_judgment — blank for user to fill
        ]
        for col_idx, val in enumerate(values, start=1):
            cell = ws_pq.cell(row=row_idx, column=col_idx, value=val)
            cell.alignment = wrap

    ws_pq.freeze_panes = "A2"
    ws_pq.auto_filter.ref = ws_pq.dimensions

    # ── Tab 3: Config ────────────────────────────────────────────────────
    ws_cfg = wb.create_sheet("Config")

    gen_model = (
        getattr(config, "TAMU_MODEL", "?")
        if getattr(config, "USE_TAMU_API", False)
        else getattr(config, "GENERATION_MODEL", "?")
    )
    cfg_rows = [
        ("experiment_name", experiment_name),
        ("date", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("git_commit", _get_git_commit()),
        ("golden_set_path", golden_set_path),
        ("n_questions", n),
        ("ragas_enabled", do_ragas),
        ("generation_model", gen_model),
        ("use_tamu_api", getattr(config, "USE_TAMU_API", False)),
    ]

    for col, label in [(1, "Key"), (2, "Value")]:
        cell = ws_cfg.cell(row=1, column=col, value=label)
        cell.font = header_font
        cell.fill = header_fill
    ws_cfg.column_dimensions["A"].width = 28
    ws_cfg.column_dimensions["B"].width = 50

    for i, (k, v) in enumerate(cfg_rows, start=2):
        ws_cfg.cell(row=i, column=1, value=k)
        ws_cfg.cell(row=i, column=2, value=str(v))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    print(f"  Excel: {output_path}")


def write_markdown(
    rows: list[BenchmarkRow],
    experiment_name: str,
    output_path: Path,
) -> None:
    n = len(rows)
    n_fn_correct = sum(1 for r in rows if r.router_function_correct)
    recall_cases = [r for r in rows if r.retrieved_correct is not None]
    recall_hits = sum(1 for r in recall_cases if r.retrieved_correct)
    citation_cases = [r for r in rows if r.router_function != "out_of_scope"]
    citation_pass = sum(1 for r in citation_cases if r.citation_pass)
    ragas_cases = [r for r in rows if r.ragas_faithfulness is not None]
    avg_faith = sum(r.ragas_faithfulness for r in ragas_cases) / len(ragas_cases) if ragas_cases else None

    def _fmt_ms(attr: str) -> str:
        v = _avg(rows, attr)
        return f"{v:.0f}" if v is not None else "N/A"

    lines = [
        f"# Benchmark Report: {experiment_name}",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Git commit:** {_get_git_commit()}  ",
        f"**Questions:** {n}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    if n:
        lines += [
            f"| Router accuracy | {n_fn_correct/n:.1%} ({n_fn_correct}/{n}) |",
            f"| Recall@k (crn+category) | {recall_hits/len(recall_cases):.1%} ({recall_hits}/{len(recall_cases)}) |"
            if recall_cases else "| Recall@k | N/A (no source_crn) |",
            f"| Citation pass rate | {citation_pass/len(citation_cases):.1%} ({citation_pass}/{len(citation_cases)}) |"
            if citation_cases else "| Citation pass rate | N/A |",
            f"| Mean RAGAS faithfulness | {avg_faith:.2f} |"
            if avg_faith is not None else "| Mean RAGAS faithfulness | not run |",
            f"| Mean total latency (ms) | {_fmt_ms('total_ms')} |",
            f"| Mean router latency (ms) | {_fmt_ms('router_ms')} |",
            f"| Mean retrieval latency (ms) | {_fmt_ms('retrieval_ms')} |",
            f"| Mean generator latency (ms) | {_fmt_ms('generator_ms')} |",
            f"| Errors | {sum(1 for r in rows if r.error)} |",
        ]

    # Per-stratum breakdown
    lines += ["", "## Router Accuracy by Stratum", "",
              "| Stratum | Correct | Total | Accuracy |",
              "|---------|---------|-------|----------|"]
    strata = sorted({r.stratum for r in rows})
    for s in strata:
        s_rows = [r for r in rows if r.stratum == s]
        s_correct = sum(1 for r in s_rows if r.router_function_correct)
        lines.append(f"| {s} | {s_correct} | {len(s_rows)} | {s_correct/len(s_rows):.0%} |")

    # Errors
    errors = [r for r in rows if r.error]
    if errors:
        lines += ["", f"## Errors ({len(errors)})", ""]
        for r in errors:
            q = r.question[:70] + "..." if len(r.question) > 70 else r.question
            lines.append(f"- **{q}** → `{r.error}`")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Markdown: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark runner — runs golden set through pipeline, exports versioned reports"
    )
    parser.add_argument("--golden-set", required=True, help="Path to golden set JSONL")
    parser.add_argument("--experiment-name", required=True,
                        help="Experiment identifier embedded in output filename (e.g. cs600_ov100)")
    parser.add_argument("--ragas", action="store_true",
                        help="Run RAGAS faithfulness/relevancy scores (~30s per question)")
    args = parser.parse_args()

    golden_path = Path(args.golden_set)
    if not golden_path.exists():
        print(f"ERROR: Golden set not found: {golden_path}")
        sys.exit(1)

    items = []
    with golden_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"\nBenchmark: {args.experiment_name}")
    print(f"Golden set: {golden_path}  ({len(items)} items)")
    if args.ragas:
        print("RAGAS: enabled")

    rows: list[BenchmarkRow] = []
    for i, item in enumerate(items, start=1):
        q_preview = item.get("question", "")[:60]
        print(f"\n[{i:3d}/{len(items)}] {q_preview}...", flush=True)
        row = run_one(item, do_ragas=args.ragas)
        rows.append(row)
        fn_ok = "v" if row.router_function_correct else "x"
        status = row.error or "OK"
        print(
            f"         fn={fn_ok} router={row.router_ms:.0f}ms "
            f"ret={row.retrieval_ms:.0f}ms gen={row.generator_ms:.0f}ms  [{status}]"
        )

    # Write reports
    ts = datetime.now().strftime("%Y%m%d")
    base_name = f"benchmark_{args.experiment_name}_{ts}"
    xlsx_path = REPORTS_DIR / f"{base_name}.xlsx"
    md_path = REPORTS_DIR / f"{base_name}.md"

    print("\nWriting reports...")
    write_excel(rows, args.experiment_name, str(golden_path), xlsx_path, args.ragas)
    write_markdown(rows, args.experiment_name, md_path)

    # Final summary
    n = len(rows)
    n_correct = sum(1 for r in rows if r.router_function_correct)
    print(f"\n{'='*55}")
    print(f"  Experiment:      {args.experiment_name}")
    print(f"  Questions:       {n}")
    if n:
        print(f"  Router accuracy: {n_correct/n:.1%} ({n_correct}/{n})")
    errors = sum(1 for r in rows if r.error)
    if errors:
        print(f"  Errors:          {errors}")
    print(f"{'='*55}")
    print(f"\nTo validate RAGAS: python evals/validate_ragas.py --benchmark {xlsx_path}")


if __name__ == "__main__":
    main()
