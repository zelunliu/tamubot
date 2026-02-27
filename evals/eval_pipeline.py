"""End-to-end pipeline evaluation harness for TamuBot.

Runs a representative test suite across all function types, captures full pipeline
traces (router → retrieval → generation), and writes two output files:
  - tamu_data/logs/eval_results_<timestamp>.jsonl  (machine-readable)
  - tamu_data/logs/eval_report_<timestamp>.md      (human-readable for Deep Research)

Usage:
    python scripts/eval_pipeline.py                    # full suite
    python scripts/eval_pipeline.py --function metadata_specific  # single function
    python scripts/eval_pipeline.py --dry-run          # router only, skip retrieval+gen
"""

import argparse
import io
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# Ensure repo root is on path when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# UTF-8 stdout for Windows (avoids cp1252 encode errors for ✅ ❌ etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from rag import router as router_mod
from rag import generator as gen_mod


# ---------------------------------------------------------------------------
# Test case definitions  — grounded in CSCE 638 / CSCE 670 (Spring 2026)
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    query: str
    function_expected: str
    description: str
    expected_course_ids: list[str] = field(default_factory=list)
    expected_specific_categories: list[str] = field(default_factory=list)
    expected_semantic_intent: bool = False
    notes: str = ""
    # Golden set provenance — used for recall@k
    source_crn: str = ""
    source_category: str = ""
    reference_answer: str = ""


TEST_SUITE: list[TestCase] = [

    # ── metadata_specific ────────────────────────────────────────────────
    # Purely factual, targeting a specific syllabus category.
    TestCase(
        query="What is the grading breakdown for CSCE 638?",
        function_expected="metadata_specific",
        description="GRADING category — exact factual lookup, high category confidence",
        expected_course_ids=["CSCE 638"],
        expected_specific_categories=["GRADING"],
        expected_semantic_intent=False,
    ),
    TestCase(
        query="What textbooks or materials are required for CSCE 670?",
        function_expected="metadata_specific",
        description="MATERIALS category — tests synonym expansion ('textbooks')",
        expected_course_ids=["CSCE 670"],
        expected_specific_categories=["MATERIALS"],
        expected_semantic_intent=False,
    ),
    TestCase(
        query="Can I use ChatGPT or AI tools in CSCE 638?",
        function_expected="metadata_specific",
        description="AI_POLICY — tests slang rewriting (ChatGPT → AI policy)",
        expected_course_ids=["CSCE 638"],
        expected_specific_categories=["AI_POLICY"],
        expected_semantic_intent=False,
    ),
    TestCase(
        query="What are the prerequisites for CSCE 638?",
        function_expected="metadata_specific",
        description="PREREQUISITES — factual prerequisite lookup",
        expected_course_ids=["CSCE 638"],
        expected_specific_categories=["PREREQUISITES"],
        expected_semantic_intent=False,
    ),
    TestCase(
        query="What is the late work policy for CSCE 670?",
        function_expected="metadata_specific",
        description="ATTENDANCE_AND_MAKEUP — 'late work' rewrite rule",
        expected_course_ids=["CSCE 670"],
        expected_specific_categories=["ATTENDANCE_AND_MAKEUP"],
        expected_semantic_intent=False,
        notes="Router rewrite rule: 'late work' → 'attendance makeup deadline extensions'",
    ),
    TestCase(
        query="When does CSCE 638 meet?",
        function_expected="metadata_specific",
        description="SCHEDULE — meeting time lookup with keyword 'when'",
        expected_course_ids=["CSCE 638"],
        expected_specific_categories=["SCHEDULE"],
        expected_semantic_intent=False,
    ),
    TestCase(
        query="Who is the instructor for CSCE 670?",
        function_expected="metadata_specific",
        description="INSTRUCTOR — instructor identity lookup",
        expected_course_ids=["CSCE 670"],
        expected_specific_categories=["INSTRUCTOR"],
        expected_semantic_intent=False,
    ),

    # ── metadata_default ─────────────────────────────────────────────────
    # Factual, no specific category — returns DEFAULT_SUMMARY_CATEGORIES.
    TestCase(
        query="Tell me about CSCE 638",
        function_expected="metadata_default",
        description="General overview request — no specific category signal",
        expected_course_ids=["CSCE 638"],
        expected_specific_categories=[],
        expected_semantic_intent=False,
    ),
    TestCase(
        query="Give me an overview of CSCE 670",
        function_expected="metadata_default",
        description="Explicit overview request — should return COURSE_OVERVIEW + PREREQUISITES + LEARNING_OUTCOMES",
        expected_course_ids=["CSCE 670"],
        expected_specific_categories=[],
        expected_semantic_intent=False,
    ),
    TestCase(
        query="What is CSCE 638 about?",
        function_expected="metadata_default",
        description="Short broad question — should trigger metadata_default not metadata_specific",
        expected_course_ids=["CSCE 638"],
        expected_specific_categories=[],
        expected_semantic_intent=False,
    ),

    # ── metadata_combined ────────────────────────────────────────────────
    # Factual, specific categories mentioned but not exclusive (specific_only=False).
    TestCase(
        query="Tell me about CSCE 638, especially the grading",
        function_expected="metadata_combined",
        description="BOUNDARY CASE: broad + specific — 'especially' signals not-exclusive focus",
        expected_course_ids=["CSCE 638"],
        expected_specific_categories=["GRADING"],
        expected_semantic_intent=False,
        notes="specific_only=False expected. Router may return metadata_specific if it reads 'especially grading' as exclusive.",
    ),
    TestCase(
        query="Give me an overview of CSCE 670, with a focus on learning outcomes",
        function_expected="metadata_combined",
        description="BOUNDARY CASE: overview + emphasis — 'with a focus on' signals combined intent",
        expected_course_ids=["CSCE 670"],
        expected_specific_categories=["LEARNING_OUTCOMES"],
        expected_semantic_intent=False,
        notes="specific_only=False expected. This is the canonical metadata_combined case.",
    ),

    # ── hybrid_specific ──────────────────────────────────────────────────
    # Specific category + advisory/subjective component (specific_only=True).
    TestCase(
        query="Is CSCE 638 strict about its AI policy?",
        function_expected="hybrid_specific",
        description="AI_POLICY + evaluative ('strict') — tests semantic_intent=True with specific category",
        expected_course_ids=["CSCE 638"],
        expected_specific_categories=["AI_POLICY"],
        expected_semantic_intent=True,
    ),
    TestCase(
        query="Is the grading for CSCE 670 fair?",
        function_expected="hybrid_specific",
        description="GRADING + evaluative ('fair') — tests advisory overlay on factual category",
        expected_course_ids=["CSCE 670"],
        expected_specific_categories=["GRADING"],
        expected_semantic_intent=True,
    ),

    # ── hybrid_default ───────────────────────────────────────────────────
    # Advisory/subjective question about a course, no specific category.
    TestCase(
        query="Is CSCE 638 a good course for machine learning research?",
        function_expected="hybrid_default",
        description="Advisory + CAREER semantic type — no specific category",
        expected_course_ids=["CSCE 638"],
        expected_specific_categories=[],
        expected_semantic_intent=True,
    ),
    TestCase(
        query="Is CSCE 670 worth taking?",
        function_expected="hybrid_default",
        description="Evaluative ('worth taking') — GENERAL semantic type, no specific category",
        expected_course_ids=["CSCE 670"],
        expected_specific_categories=[],
        expected_semantic_intent=True,
    ),
    TestCase(
        query="How hard is CSCE 638?",
        function_expected="hybrid_default",
        description="DIFFICULTY semantic type — broad difficulty question, no specific category",
        expected_course_ids=["CSCE 638"],
        expected_specific_categories=[],
        expected_semantic_intent=True,
    ),

    # ── hybrid_specific (explicit-category advisory) ─────────────────────
    # The category is explicitly named as the subject ("based on the grading structure",
    # "given its learning outcomes") — router correctly sets specific_only=True.
    # hybrid_specific = factual category retrieval + semantic advisory overlay.
    TestCase(
        query="Does CSCE 638 have a heavy workload based on the grading structure?",
        function_expected="hybrid_specific",
        description="DIFFICULTY + GRADING — 'based on the grading structure' names the category explicitly → specific_only=True → hybrid_specific",
        expected_course_ids=["CSCE 638"],
        expected_specific_categories=["GRADING"],
        expected_semantic_intent=True,
        notes="Router correctly sets specific_only=True: the grading structure is the explicit subject, not just background context. hybrid_specific fetches GRADING + applies DIFFICULTY overlay.",
    ),
    TestCase(
        query="Is CSCE 670 good preparation for a PhD in information retrieval, given its learning outcomes?",
        function_expected="hybrid_specific",
        description="CAREER + LEARNING_OUTCOMES — 'given its learning outcomes' names the category explicitly → specific_only=True → hybrid_specific",
        expected_course_ids=["CSCE 670"],
        expected_specific_categories=["LEARNING_OUTCOMES"],
        expected_semantic_intent=True,
        notes="Router correctly sets specific_only=True: learning outcomes are the explicit subject of evaluation. hybrid_specific fetches LEARNING_OUTCOMES + applies CAREER overlay.",
    ),

    # ── metadata_specific (multi-course comparisons) ──────────────────────
    # Factual comparisons across two courses — same function, parallel fetch.
    TestCase(
        query="Compare the AI policies of CSCE 638 and CSCE 670",
        function_expected="metadata_specific",
        description="Multi-course AI_POLICY comparison — factual side-by-side",
        expected_course_ids=["CSCE 638", "CSCE 670"],
        expected_specific_categories=["AI_POLICY"],
        expected_semantic_intent=False,
    ),
    TestCase(
        query="What are the grading differences between CSCE 638 and CSCE 670?",
        function_expected="metadata_specific",
        description="Multi-course GRADING comparison — tests multi-course parallel fetch",
        expected_course_ids=["CSCE 638", "CSCE 670"],
        expected_specific_categories=["GRADING"],
        expected_semantic_intent=False,
    ),
    TestCase(
        query="Compare CSCE 638 and CSCE 670 prerequisites",
        function_expected="metadata_specific",
        description="Multi-course PREREQUISITES — tests prerequisite extraction across courses",
        expected_course_ids=["CSCE 638", "CSCE 670"],
        expected_specific_categories=["PREREQUISITES"],
        expected_semantic_intent=False,
    ),

    # ── hybrid_default (multi-course advisory) ────────────────────────────
    TestCase(
        query="Is CSCE 638 harder than CSCE 670?",
        function_expected="hybrid_default",
        description="Multi-course DIFFICULTY comparison — advisory, no specific category",
        expected_course_ids=["CSCE 638", "CSCE 670"],
        expected_specific_categories=[],
        expected_semantic_intent=True,
        notes="Router should extract both course IDs and semantic_intent=True, semantic_type=DIFFICULTY",
    ),
    TestCase(
        query="Which course, CSCE 638 or CSCE 670, is better for an ML career?",
        function_expected="hybrid_default",
        description="Multi-course CAREER advisory — tests CAREER semantic type extraction",
        expected_course_ids=["CSCE 638", "CSCE 670"],
        expected_specific_categories=[],
        expected_semantic_intent=True,
        notes="semantic_type=CAREER expected",
    ),
    TestCase(
        query="Should I take CSCE 638 or CSCE 670 first?",
        function_expected="hybrid_default",
        description="Multi-course PLANNING advisory — course sequencing question",
        expected_course_ids=["CSCE 638", "CSCE 670"],
        expected_specific_categories=[],
        expected_semantic_intent=True,
        notes="semantic_type=PLANNING expected",
    ),

    # ── semantic_general ─────────────────────────────────────────────────
    # No course ID, advisory/discovery — searches full corpus.
    TestCase(
        query="Which courses will help me become an AI engineer?",
        function_expected="semantic_general",
        description="CAREER discovery — no course ID, broad advisory",
        expected_course_ids=[],
        expected_semantic_intent=True,
        notes="semantic_type=CAREER expected",
    ),
    TestCase(
        query="What is the academic integrity policy at Texas A&M?",
        function_expected="semantic_general",
        description="University policy query — no course ID; treated as broad academic discovery",
        expected_course_ids=[],
        expected_semantic_intent=True,
        notes="Policies have no course_id. semantic_type=ACADEMIC expected.",
    ),
    TestCase(
        query="Is there a computer science course about machine learning at TAMU?",
        function_expected="semantic_general",
        description="PLANNING discovery — no specific course ID, seeks course recommendations",
        expected_course_ids=[],
        expected_semantic_intent=True,
        notes="semantic_type=PLANNING expected",
    ),
    TestCase(
        query="What accommodations are available for students with disabilities at TAMU?",
        function_expected="semantic_general",
        description="ADA policy — no course ID, broad academic resource query",
        expected_course_ids=[],
        expected_semantic_intent=True,
        notes="semantic_type=ACADEMIC expected",
    ),
    TestCase(
        query="How should I plan my CS coursework for a research career?",
        function_expected="semantic_general",
        description="PLANNING advisory — no course ID, high-level academic planning question",
        expected_course_ids=[],
        expected_semantic_intent=True,
        notes="semantic_type=PLANNING expected",
    ),

    # ── out_of_scope ─────────────────────────────────────────────────────
    TestCase(
        query="What's the weather in College Station today?",
        function_expected="out_of_scope",
        description="Classic off-topic — should return canned response immediately",
        expected_semantic_intent=False,
    ),
    TestCase(
        query="Can you write a cover letter for a Google internship?",
        function_expected="out_of_scope",
        description="Non-academic task — off-topic, no course IDs, no semantic intent",
        expected_semantic_intent=False,
    ),
    TestCase(
        query="What are the best restaurants near Texas A&M campus?",
        function_expected="out_of_scope",
        description="Local info — clearly off-topic",
        expected_semantic_intent=False,
    ),
    TestCase(
        query="Howdy! What can you help me with?",
        function_expected="out_of_scope",
        description="Greeting — tests that greetings don't trigger retrieval",
        expected_semantic_intent=False,
        notes="Borderline: could be semantic_general if router sees 'help me' as advisory",
    ),
]


# ---------------------------------------------------------------------------
# Result capture
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    # Identity
    test_id: int
    query: str
    description: str
    notes: str

    # Router — expected
    function_expected: str
    course_ids_expected: list[str]
    specific_categories_expected: list[str]
    semantic_intent_expected: bool

    # Router — actual
    function_actual: str
    course_ids_extracted: list[str]
    specific_categories_extracted: list[str]
    semantic_intent_actual: bool
    semantic_type_actual: str | None
    category_confidence: float
    retrieval_mode_actual: str
    rewritten_query: str

    # Router — correctness
    function_correct: bool
    course_ids_correct: bool
    specific_categories_correct: bool   # expected ⊆ extracted

    # Retrieval
    chunks_retrieved: int
    unique_courses_in_results: list[str]
    unique_categories_in_results: list[str]
    retrieval_error: str | None

    # Generation
    response_preview: str               # first 400 chars
    response_length_chars: int
    has_citations: bool
    citation_count: int
    generation_skipped: bool            # True for out_of_scope or dry-run
    generation_error: str | None

    # Retrieval quality
    recall_hit: bool | None         # True if source chunk (crn+category) found; None if no source_crn

    # Answer quality (RAGAS — only populated when --ragas flag is set)
    ragas_faithfulness: float | None
    ragas_answer_relevancy: float | None

    # Timing (ms)
    latency_router_ms: float
    latency_retrieval_ms: float
    latency_generation_ms: float
    latency_ragas_ms: float
    latency_total_ms: float


def _count_citations(text: str) -> int:
    return len(re.findall(r"\[Source \d+\]", text))


# ---------------------------------------------------------------------------
# Run a single test case through the full pipeline
# ---------------------------------------------------------------------------

def run_test(
    tc: TestCase,
    test_id: int,
    dry_run: bool = False,
    do_ragas: bool = False,
) -> EvalResult:
    t0 = time.perf_counter()

    # ── Stage 1: Router ──────────────────────────────────────────────────
    t_router_start = time.perf_counter()
    try:
        rr = router_mod.classify_query(tc.query)
        router_error = None
    except Exception as e:
        rr = router_mod.RouterResult(rewritten_query=tc.query, category_confidence=0.0)
        router_error = str(e)
    latency_router_ms = (time.perf_counter() - t_router_start) * 1000

    # ── Stage 2: Retrieval + Rerank ──────────────────────────────────────
    t_retrieval_start = time.perf_counter()
    retrieval_error = None
    chunks: list[dict] = []

    if dry_run or not rr.requires_retrieval:
        pass
    else:
        try:
            chunks = _do_retrieval(rr, tc.query)
        except Exception as e:
            retrieval_error = str(e)

    latency_retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000

    # ── Stage 3: Generation ───────────────────────────────────────────────
    t_gen_start = time.perf_counter()
    generation_error = None
    response_text = ""
    generation_skipped = dry_run or (rr.function == "out_of_scope")

    if generation_skipped and rr.function == "out_of_scope":
        response_text = gen_mod.generate([], tc.query, function="out_of_scope")
    elif not dry_run and not retrieval_error:
        try:
            response_text = gen_mod.generate(
                chunks, tc.query,
                function=rr.function,
                course_ids=rr.course_ids,
                semantic_type=rr.semantic_type,
            )
        except Exception as e:
            generation_error = str(e)

    latency_generation_ms = (time.perf_counter() - t_gen_start) * 1000

    # ── Recall@k — did the source chunk surface in retrieved results? ─────
    # source_crn is only populated when loading from a golden set JSONL.
    # For the built-in TEST_SUITE (no source_crn), recall_hit is None.
    recall_hit: bool | None = None
    if tc.source_crn:
        recall_hit = any(
            c.get("crn") == tc.source_crn and c.get("category") == tc.source_category
            for c in chunks
        )

    # ── RAGAS quality scores (optional — requires --ragas flag) ───────────
    t_ragas_start = time.perf_counter()
    ragas_faithfulness: float | None = None
    ragas_answer_relevancy: float | None = None

    if do_ragas and not dry_run and chunks and response_text and not generation_error:
        try:
            from rag.observability import compute_ragas_metrics
            contexts = [c.get("content", "") for c in chunks if c.get("content")]
            scores = compute_ragas_metrics(
                question=tc.query,
                contexts=contexts,
                answer=response_text,
            )
            ragas_faithfulness = scores.get("faithfulness")
            ragas_answer_relevancy = scores.get("answer_relevancy")
        except Exception as e:
            pass  # RAGAS failure is non-fatal

    latency_ragas_ms = (time.perf_counter() - t_ragas_start) * 1000
    latency_total_ms = (time.perf_counter() - t0) * 1000

    # ── Routing correctness ───────────────────────────────────────────────
    unique_courses = sorted({d.get("course_id", "") for d in chunks if d.get("course_id")})
    unique_categories = sorted({d.get("category", "") for d in chunks if d.get("category")})

    cids_correct = (
        set(tc.expected_course_ids) <= set(rr.course_ids)
        if tc.expected_course_ids else True
    )
    cats_correct = (
        set(tc.expected_specific_categories) <= set(rr.specific_categories)
        if tc.expected_specific_categories else True
    )

    return EvalResult(
        test_id=test_id,
        query=tc.query,
        description=tc.description,
        notes=tc.notes,
        function_expected=tc.function_expected,
        course_ids_expected=tc.expected_course_ids,
        specific_categories_expected=tc.expected_specific_categories,
        semantic_intent_expected=tc.expected_semantic_intent,
        function_actual=rr.function,
        course_ids_extracted=rr.course_ids,
        specific_categories_extracted=rr.specific_categories,
        semantic_intent_actual=rr.semantic_intent,
        semantic_type_actual=rr.semantic_type,
        category_confidence=round(rr.category_confidence, 3),
        retrieval_mode_actual=rr.retrieval_mode,
        rewritten_query=rr.rewritten_query,
        function_correct=(rr.function == tc.function_expected),
        course_ids_correct=cids_correct,
        specific_categories_correct=cats_correct,
        chunks_retrieved=len(chunks),
        unique_courses_in_results=unique_courses,
        unique_categories_in_results=unique_categories,
        retrieval_error=retrieval_error,
        response_preview=(response_text[:400] + "..." if len(response_text) > 400 else response_text),
        response_length_chars=len(response_text),
        has_citations=bool(re.search(r"\[Source \d+\]", response_text)),
        citation_count=_count_citations(response_text),
        generation_skipped=generation_skipped,
        generation_error=generation_error,
        recall_hit=recall_hit,
        ragas_faithfulness=ragas_faithfulness,
        ragas_answer_relevancy=ragas_answer_relevancy,
        latency_router_ms=round(latency_router_ms, 1),
        latency_retrieval_ms=round(latency_retrieval_ms, 1),
        latency_generation_ms=round(latency_generation_ms, 1),
        latency_ragas_ms=round(latency_ragas_ms, 1),
        latency_total_ms=round(latency_total_ms, 1),
    )


def _do_retrieval(rr: router_mod.RouterResult, query: str) -> list[dict]:
    """Re-execute only the retrieval+rerank stages given a pre-classified RouterResult.

    Mirrors the logic in router._retrieve_and_rerank() for use in the eval harness
    (avoids re-running the router LLM call).
    """
    from rag import search, reranker

    search_query = rr.rewritten_query or query

    if not rr.requires_retrieval:
        return []

    fn = rr.function
    mode = rr.retrieval_mode
    dk = router_mod._compute_dynamic_k(fn, len(rr.course_ids))
    retrieve_k = dk["retrieve_k"]
    rerank_k = dk["rerank_k"]

    course_ids = rr.course_ids
    specific_cats = rr.specific_categories

    # semantic_general
    if fn == "semantic_general":
        results = search.search_semantic(search_query, top_k=retrieve_k)
        reranked = reranker.rerank(search_query, results, top_k=rerank_k)
        return router_mod._deduplicate_chunks(reranked)

    if fn == "out_of_scope" or not course_ids:
        return []

    # Determine categories
    if fn in ("metadata_default", "hybrid_default"):
        categories = config.DEFAULT_SUMMARY_CATEGORIES
    elif fn in ("metadata_specific", "hybrid_specific"):
        categories = specific_cats or config.DEFAULT_SUMMARY_CATEGORIES
    elif fn in ("metadata_combined", "hybrid_combined"):
        seen_cats: set[str] = set()
        categories = []
        for c in (config.DEFAULT_SUMMARY_CATEGORIES + specific_cats):
            if c not in seen_cats:
                seen_cats.add(c)
                categories.append(c)
    else:
        categories = config.DEFAULT_SUMMARY_CATEGORIES

    def fetch_course(cid: str) -> list[dict]:
        if mode == "metadata":
            return search.search_by_course_categories(cid, categories)
        filters: dict = {"course_id": cid}
        return search.hybrid_search(search_query, filters=filters, k=retrieve_k)

    if len(course_ids) == 1:
        results = fetch_course(course_ids[0])
        if mode == "metadata":
            return router_mod._deduplicate_chunks(results)
        reranked = reranker.rerank(search_query, results, top_k=rerank_k)
        return router_mod._deduplicate_chunks(reranked)

    # Multi-course
    course_groups: dict[str, list[dict]] = {cid: fetch_course(cid) for cid in course_ids}

    if mode == "metadata":
        combined: list[dict] = []
        for cid in course_ids:
            combined.extend(course_groups[cid])
        return router_mod._deduplicate_chunks(combined)

    reranked = reranker.rerank_multi_course(search_query, course_groups, top_k_per_course=rerank_k)
    return router_mod._deduplicate_chunks(reranked)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _pct(val: float) -> str:
    return f"{val:.0%}"


def write_jsonl(results: list[EvalResult], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


def write_markdown_report(
    results: list[EvalResult],
    path: Path,
    run_ts: str,
    dry_run: bool,
) -> None:
    """Generate a comprehensive markdown report for human/Gemini Deep Research analysis."""

    functions = [
        "metadata_default", "metadata_specific", "metadata_combined",
        "hybrid_default", "hybrid_specific", "hybrid_combined",
        "semantic_general", "out_of_scope",
    ]

    # ── Aggregate stats ────────────────────────────────────────────────
    total = len(results)
    fn_correct = sum(1 for r in results if r.function_correct)
    cids_correct = sum(1 for r in results if r.course_ids_correct)
    cats_correct = sum(1 for r in results if r.specific_categories_correct)
    with_errors = [r for r in results if r.retrieval_error or r.generation_error]
    with_citations = [r for r in results if r.has_citations and not r.generation_skipped]
    generated = [r for r in results if not r.generation_skipped]

    # Recall@k (only cases with source_crn populated)
    recall_cases = [r for r in results if r.recall_hit is not None]
    recall_hits = sum(1 for r in recall_cases if r.recall_hit)

    # RAGAS (only cases where scores were computed)
    ragas_cases = [r for r in results if r.ragas_faithfulness is not None]
    avg_faithfulness = (
        sum(r.ragas_faithfulness for r in ragas_cases) / len(ragas_cases)
        if ragas_cases else None
    )
    ragas_rel_cases = [r for r in results if r.ragas_answer_relevancy is not None]
    avg_relevancy = (
        sum(r.ragas_answer_relevancy for r in ragas_rel_cases) / len(ragas_rel_cases)
        if ragas_rel_cases else None
    )

    avg_total_ms = sum(r.latency_total_ms for r in results) / total if total else 0
    avg_router_ms = sum(r.latency_router_ms for r in results) / total if total else 0
    avg_chunks = sum(r.chunks_retrieved for r in results) / total if total else 0

    lines: list[str] = []

    # ── Header ─────────────────────────────────────────────────────────
    lines += [
        "# TamuBot Pipeline Evaluation Report",
        "",
        f"**Run timestamp:** {run_ts}  ",
        f"**Mode:** {'Dry-run (router only)' if dry_run else 'Full pipeline'}  ",
        f"**Router model:** `{config.MODEL_NAME}`  ",
        f"**Generator model:** `{config.GENERATION_MODEL}`  ",
        f"**Voyage rerank model:** `{config.VOYAGE_RERANK_MODEL}`  ",
        f"**RETRIEVAL_TOP_K:** {config.RETRIEVAL_TOP_K}  ",
        f"**RERANK_TOP_K:** {config.RERANK_TOP_K}  ",
        f"**CATEGORY_CONFIDENCE_THRESHOLD:** {config.CATEGORY_CONFIDENCE_THRESHOLD}  ",
        "",
        "---",
        "",
    ]

    # ── Function taxonomy ──────────────────────────────────────────────
    lines += [
        "## Function Taxonomy & Derivation Matrix",
        "",
        "Functions are derived mechanically from extracted variables (no intent classification).",
        "",
        "| course_ids | semantic_intent | specific_categories | specific_only | function |",
        "|---|---|---|---|---|",
        "| empty | True | any | any | `semantic_general` |",
        "| empty | False | any | any | `out_of_scope` |",
        "| present | False | empty | — | `metadata_default` |",
        "| present | False | populated | True | `metadata_specific` |",
        "| present | False | populated | False | `metadata_combined` |",
        "| present | True | empty | — | `hybrid_default` |",
        "| present | True | populated | True | `hybrid_specific` |",
        "| present | True | populated | False | `hybrid_combined` |",
        "",
        "**Retrieval mode derivation:** `metadata` if category_confidence ≥ "
        f"{config.CATEGORY_CONFIDENCE_THRESHOLD}, else `hybrid`; `semantic` if no course_ids.",
        "",
        "**No reranking on metadata path** — exact index lookups, already correct.",
        "**Reranking applies on hybrid and semantic paths.**",
        "",
        "---",
        "",
    ]

    # ── Executive summary ─────────────────────────────────────────────
    recall_str = (
        f"{recall_hits}/{len(recall_cases)} ({_pct(recall_hits/len(recall_cases))})"
        if recall_cases else "N/A (no source_crn — built-in suite)"
    )
    faith_str = f"{avg_faithfulness:.3f}" if avg_faithfulness is not None else "not run (omit --ragas)"
    rel_str = f"{avg_relevancy:.3f}" if avg_relevancy is not None else "not run (omit --ragas)"

    lines += [
        "## Executive Summary",
        "",
        f"- **Total test cases:** {total}",
        f"- **Function accuracy:** {fn_correct}/{total} ({_pct(fn_correct/total if total else 0)})",
        f"- **Course ID extraction accuracy:** {cids_correct}/{total} ({_pct(cids_correct/total if total else 0)})",
        f"- **Category extraction accuracy:** {cats_correct}/{total} ({_pct(cats_correct/total if total else 0)})",
        f"- **Retrieval/generation errors:** {len(with_errors)}",
        f"- **Recall@k (source chunk in results):** {recall_str}",
        f"- **RAGAS Faithfulness (avg):** {faith_str}",
        f"- **RAGAS Answer Relevancy (avg):** {rel_str}",
        f"- **Responses with citations:** {len(with_citations)}/{len(generated)} ({_pct(len(with_citations)/len(generated) if generated else 0)})",
        f"- **Avg total latency:** {avg_total_ms:.0f} ms",
        f"- **Avg router latency:** {avg_router_ms:.0f} ms",
        f"- **Avg chunks retrieved (post-dedup):** {avg_chunks:.1f}",
        "",
        "---",
        "",
    ]

    # ── Per-function summary table ─────────────────────────────────────
    lines += [
        "## Results by Function",
        "",
        "| Function | Tests | Fn Acc | CID Acc | Cat Acc | Recall@k | Avg Chunks | Avg Latency (ms) | Errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for fn in functions:
        group = [r for r in results if r.function_expected == fn]
        if not group:
            continue
        n = len(group)
        fc = sum(1 for r in group if r.function_correct)
        cc = sum(1 for r in group if r.course_ids_correct)
        sc = sum(1 for r in group if r.specific_categories_correct)
        ac = sum(r.chunks_retrieved for r in group) / n
        al = sum(r.latency_total_ms for r in group) / n
        errs = sum(1 for r in group if r.retrieval_error or r.generation_error)
        rg = [r for r in group if r.recall_hit is not None]
        rk_str = f"{sum(1 for r in rg if r.recall_hit)}/{len(rg)}" if rg else "—"
        lines.append(
            f"| `{fn}` | {n} | {fc}/{n} ({_pct(fc/n)}) | {cc}/{n} ({_pct(cc/n)}) | "
            f"{sc}/{n} ({_pct(sc/n)}) | {rk_str} | {ac:.1f} | {al:.0f} | {errs} |"
        )
    lines += ["", "---", ""]

    # ── Per-query detailed results ────────────────────────────────────
    lines += ["## Detailed Results by Function", ""]

    for fn in functions:
        group = [r for r in results if r.function_expected == fn]
        if not group:
            continue
        lines += [f"### {fn}", ""]
        for r in group:
            status = "✅" if r.function_correct else "❌"
            lines += [
                f"#### Test {r.test_id}: {status} `{r.query}`",
                "",
                f"**Description:** {r.description}",
            ]
            if r.notes:
                lines.append(f"**Notes:** {r.notes}")
            lines += [
                "",
                f"| Field | Value |",
                f"|---|---|",
                f"| Function expected | `{r.function_expected}` |",
                f"| Function actual | `{r.function_actual}` |",
                f"| Retrieval mode | `{r.retrieval_mode_actual}` |",
                f"| Category confidence | {r.category_confidence} |",
                f"| Course IDs expected | `{r.course_ids_expected}` |",
                f"| Course IDs extracted | `{r.course_ids_extracted}` |",
                f"| Categories expected | `{r.specific_categories_expected}` |",
                f"| Categories extracted | `{r.specific_categories_extracted}` |",
                f"| Semantic intent expected | `{r.semantic_intent_expected}` |",
                f"| Semantic intent actual | `{r.semantic_intent_actual}` |",
                f"| Semantic type | `{r.semantic_type_actual}` |",
                f"| Rewritten query | _{r.rewritten_query}_ |",
                f"| Chunks retrieved (post-dedup) | {r.chunks_retrieved} |",
                f"| Unique courses in results | `{r.unique_courses_in_results}` |",
                f"| Unique categories in results | `{r.unique_categories_in_results}` |",
                f"| Recall@k (source chunk found) | {'✅' if r.recall_hit else ('❌' if r.recall_hit is False else '—')} |",
                f"| RAGAS Faithfulness | {f'{r.ragas_faithfulness:.3f}' if r.ragas_faithfulness is not None else '—'} |",
                f"| RAGAS Answer Relevancy | {f'{r.ragas_answer_relevancy:.3f}' if r.ragas_answer_relevancy is not None else '—'} |",
                f"| Latency: router | {r.latency_router_ms} ms |",
                f"| Latency: retrieval+rerank | {r.latency_retrieval_ms} ms |",
                f"| Latency: generation | {r.latency_generation_ms} ms |",
                f"| Latency: RAGAS | {r.latency_ragas_ms} ms |",
                f"| Latency: total | {r.latency_total_ms} ms |",
                f"| Citations in response | {r.citation_count} (`[Source N]` pattern) |",
                f"| Response length | {r.response_length_chars} chars |",
            ]
            if r.retrieval_error:
                lines.append(f"| Retrieval error | `{r.retrieval_error}` |")
            if r.generation_error:
                lines.append(f"| Generation error | `{r.generation_error}` |")
            lines += [
                "",
                "**Response preview:**",
                "",
                "```",
                r.response_preview if r.response_preview else "(empty — see error above)",
                "```",
                "",
            ]

    # ── Aggregate latency breakdown ────────────────────────────────────
    lines += [
        "---",
        "",
        "## Latency Breakdown by Function",
        "",
        "| Function | Router (ms) | Retrieval (ms) | Generation (ms) | Total (ms) |",
        "|---|---|---|---|---|",
    ]
    for fn in functions:
        group = [r for r in results if r.function_expected == fn]
        if not group:
            continue
        n = len(group)
        ar = sum(r.latency_router_ms for r in group) / n
        at = sum(r.latency_retrieval_ms for r in group) / n
        ag = sum(r.latency_generation_ms for r in group) / n
        al = sum(r.latency_total_ms for r in group) / n
        lines.append(f"| `{fn}` | {ar:.0f} | {at:.0f} | {ag:.0f} | {al:.0f} |")
    lines += ["", "---", ""]

    # ── Error log ─────────────────────────────────────────────────────
    if with_errors:
        lines += ["## Errors Encountered", ""]
        for r in with_errors:
            lines += [
                f"- **Test {r.test_id}** (`{r.function_expected}`): `{r.query}`",
            ]
            if r.retrieval_error:
                lines.append(f"  - Retrieval: `{r.retrieval_error}`")
            if r.generation_error:
                lines.append(f"  - Generation: `{r.generation_error}`")
        lines += ["", "---", ""]

    # ── Design notes ──────────────────────────────────────────────────
    lines += [
        "## Design Notes & Known Boundary Cases",
        "",
        "### 1. metadata_combined vs metadata_specific Boundary",
        "The `specific_only` flag distinguishes these two functions.",
        "A query like 'Tell me about CSCE 638, especially the grading' should yield",
        "`specific_only=False` (metadata_combined) because it requests a general overview",
        "with emphasis, not ONLY grading. The router may misclassify this as `metadata_specific`",
        "if it interprets 'especially grading' as exclusive focus.",
        "**Monitor:** category_confidence and specific_only extraction accuracy.",
        "",
        "### 2. Policy Queries Without Course IDs → semantic_general",
        "Under the new schema, queries like 'What is the academic integrity policy?'",
        "have no course_ids, so they route to `semantic_general` (semantic_intent=True,",
        "semantic_type=ACADEMIC). This uses `search_semantic` over the full corpus,",
        "which should surface UNIVERSITY_POLICIES chunks. This is the correct behavior.",
        "",
        "### 3. Multi-Course Advisory Queries",
        "Queries like 'Is CSCE 638 harder than CSCE 670?' extract both course_ids and",
        "semantic_intent=True, routing to `hybrid_default` with parallel per-course fetches.",
        "The generator receives balanced context from both courses and applies the",
        "DIFFICULTY semantic type overlay.",
        "",
        "### 4. category_confidence Threshold",
        f"The threshold is {config.CATEGORY_CONFIDENCE_THRESHOLD}. When the router is uncertain",
        "about category extraction (e.g. ambiguous questions), retrieval_mode falls back",
        "to 'hybrid', running hybrid_search without a category filter.",
        "This prevents incorrect category filters from missing relevant chunks.",
        "",
        "### 5. No Reranking on Metadata Path",
        "When retrieval_mode='metadata', `search_by_course_categories` returns exact",
        "index lookups sorted by category order. Reranking is skipped because the chunks",
        "are already the correct ones — no need to score against the query.",
        "",
        "---",
        "",
        "## Questions for Deep Research Analysis",
        "",
        "1. **Function derivation accuracy**: Are the 7 functions (+ out_of_scope) sufficient",
        "   to cover the full range of queries a real TAMU student would ask?",
        "",
        "2. **specific_only boundary**: How reliably does the router extract `specific_only=True`",
        "   vs `False`? What prompt changes would improve this distinction?",
        "",
        "3. **category_confidence calibration**: Is 0.7 the right threshold? Does it correctly",
        "   separate cases where category filtering helps vs. hurts retrieval?",
        "",
        "4. **semantic_intent vs factual boundary**: Which queries are hardest to classify?",
        "   Where does the router confuse evaluative language with factual requests?",
        "",
        "5. **Multi-course handling**: For comparison queries, does the parallel per-course",
        "   fetch + rerank_multi_course produce balanced, representative results?",
        "",
        "6. **semantic_general retrieval quality**: For queries with no course_ids, does",
        "   search_semantic over the full corpus return relevant chunks?",
        "   What is the false-positive rate for out-of-scope topics?",
        "",
        "7. **Retrieval quality signals**: Given chunk counts and unique categories in results,",
        "   are we over- or under-retrieving for each function type?",
        "",
        "8. **Latency budget**: How should latency be distributed across router / retrieval /",
        "   generation for a good UX? Which function types are slowest?",
        "",
        "9. **Missing query types**: What student queries are NOT well-served by any of the",
        "   7 functions and would result in degraded responses?",
        "",
        "10. **Production readiness**: Based on these results, what is the overall readiness",
        "    of the system? What are the top 3 highest-priority improvements?",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_golden_set(path: Path) -> list[TestCase]:
    """Load a golden set JSONL (from generate_golden_set.py) as TestCase objects."""
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cases.append(TestCase(
                query=d.get("question", d.get("query", "")),
                function_expected=d.get("expected_function", ""),
                description=d.get("stratum", ""),
                expected_course_ids=d.get("expected_course_ids", []),
                expected_specific_categories=d.get("expected_specific_categories", []),
                expected_semantic_intent=d.get("expected_semantic_intent", False),
                notes=d.get("source_category", ""),
                source_crn=d.get("source_crn", "") or "",
                source_category=d.get("source_category", "") or "",
                reference_answer=d.get("reference_answer", "") or "",
            ))
    return cases


def main():
    parser = argparse.ArgumentParser(description="TamuBot pipeline evaluation harness")
    parser.add_argument("--function", help="Run only tests for this function type")
    parser.add_argument("--dry-run", action="store_true",
                        help="Router classification only — skip retrieval and generation")
    parser.add_argument("--test-id", type=int, help="Run only this specific test ID (1-indexed)")
    parser.add_argument("--golden-set", metavar="PATH",
                        help="Path to golden set JSONL (from generate_golden_set.py). "
                             "Runs those questions instead of the built-in TEST_SUITE.")
    parser.add_argument("--ragas", action="store_true",
                        help="Run RAGAS Faithfulness + AnswerRelevancy scoring on each "
                             "generated response. Adds ~1 Gemini call per question.")
    parser.add_argument("--output-dir", default="tamu_data/logs",
                        help="Directory for output files (default: tamu_data/logs)")
    args = parser.parse_args()

    # Load test suite — golden set JSONL or built-in
    if args.golden_set:
        gs_path = Path(args.golden_set)
        if not gs_path.exists():
            print(f"Golden set not found: {gs_path}")
            sys.exit(1)
        base_suite = _load_golden_set(gs_path)
        print(f"Loaded {len(base_suite)} test cases from {gs_path.name}")
    else:
        base_suite = TEST_SUITE

    # Filter test suite
    suite = base_suite
    if args.function:
        suite = [tc for tc in suite if tc.function_expected == args.function]
        if not suite:
            print(f"No tests found for function: {args.function}")
            valid = sorted(set(tc.function_expected for tc in base_suite))
            print(f"Valid functions: {', '.join(valid)}")
            sys.exit(1)
    if args.test_id:
        idx = args.test_id - 1
        if not (0 <= idx < len(base_suite)):
            print(f"--test-id must be between 1 and {len(base_suite)}")
            sys.exit(1)
        suite = [base_suite[idx]]

    # Prepare output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    jsonl_path = output_dir / f"eval_results_{run_ts}.jsonl"
    md_path = output_dir / f"eval_report_{run_ts}.md"

    print(f"\nTamuBot Pipeline Evaluation — {run_ts}")
    print(f"Mode: {'DRY-RUN (router only)' if args.dry_run else 'Full pipeline'}"
          f"{' + RAGAS' if args.ragas else ''}")
    print(f"Tests: {len(suite)}")
    print(f"Output: {jsonl_path.name} + {md_path.name}")
    print("-" * 60)

    results: list[EvalResult] = []
    for i, tc in enumerate(suite, 1):
        test_id = args.test_id if args.test_id else i
        fn_label = f"[{tc.function_expected}]".ljust(22)
        print(f"  {i:2d}/{len(suite)} {fn_label} {tc.query[:60]}...")
        sys.stdout.flush()

        r = run_test(tc, test_id=test_id, dry_run=args.dry_run, do_ragas=args.ragas)
        results.append(r)

        tick = "[OK]" if r.function_correct else "[FAIL]"
        err = f" ERR: {r.retrieval_error or r.generation_error}" if (r.retrieval_error or r.generation_error) else ""
        recall_tag = "" if r.recall_hit is None else (f" recall={'HIT' if r.recall_hit else 'MISS'}")
        ragas_tag = (
            f" faith={r.ragas_faithfulness:.2f} rel={r.ragas_answer_relevancy:.2f}"
            if r.ragas_faithfulness is not None else ""
        )
        print(
            f"       {tick} fn={r.function_actual} mode={r.retrieval_mode_actual} "
            f"catconf={r.category_confidence:.2f} chunks={r.chunks_retrieved}"
            f"{recall_tag}{ragas_tag} {r.latency_total_ms:.0f}ms{err}"
        )

    print("-" * 60)
    fn_acc = sum(1 for r in results if r.function_correct) / len(results)
    print(f"Function accuracy: {fn_acc:.0%}  ({sum(1 for r in results if r.function_correct)}/{len(results)})")
    rc = [r for r in results if r.recall_hit is not None]
    if rc:
        rh = sum(1 for r in rc if r.recall_hit)
        print(f"Recall@k:          {rh/len(rc):.0%}  ({rh}/{len(rc)} source chunks found)")
    rf = [r for r in results if r.ragas_faithfulness is not None]
    if rf:
        print(f"RAGAS Faithfulness:   {sum(r.ragas_faithfulness for r in rf)/len(rf):.3f}  (avg over {len(rf)} cases)")
    rr2 = [r for r in results if r.ragas_answer_relevancy is not None]
    if rr2:
        print(f"RAGAS Ans Relevancy:  {sum(r.ragas_answer_relevancy for r in rr2)/len(rr2):.3f}  (avg over {len(rr2)} cases)")

    # Write outputs
    write_jsonl(results, jsonl_path)
    write_markdown_report(results, md_path, run_ts, dry_run=args.dry_run)
    print(f"\nResults -> {jsonl_path}")
    print(f"Report  -> {md_path}")
    print("\nSend the .md report to Gemini Deep Research for analysis.")


if __name__ == "__main__":
    main()
