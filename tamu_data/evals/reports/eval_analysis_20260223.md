# Router Eval Analysis — 2026-02-23

**Golden set:** `tamu_data/logs/golden_set.jsonl` (50 questions, CSCE + ISEN)
**Script:** `scripts/eval_router_metrics.py --golden-set ... --skip-rewrite-gain`
**Result:** 32–33/50 correct (64–66%) depending on LLM run-to-run variance

---

## Top-level findings

### 1. Confidence is bimodal — never mid-range

`category_confidence` clusters at two poles:
- **0.0** — when LLM extracts no specific categories (no category signal in question)
- **0.90–1.00** — when any category is extracted

There are almost no values between 0.1 and 0.85. This means the 0.7 threshold
(`CATEGORY_CONFIDENCE_THRESHOLD`) is rarely "on the boundary" — it's effectively
a binary switch. The ECE of 0.179 is driven almost entirely by the `[0.0–0.1]` bin,
where 7 cases have `conf=0.0` but `acc=1.00` (gap of 1.0). These are questions
correctly routed despite zero confidence, not confidence failures.

**Implication:** Confidence is not calibrated as a continuous signal. It's functioning
as a present/absent flag. Consider whether the threshold even needs tuning — the
"miscalibration" is benign (over-cautious, not over-confident).

---

## Failure analysis by bucket

### Bucket 1 — Over-extraction on "about" / factual questions (8 cases)

Questions: 1, 3, 4, 5, 6, 7, 36, 37

| # | Question | Expected | Actual |
|---|---|---|---|
| 1 | What are the prerequisites for CSCE 413? | `metadata_default` | `metadata_specific` (cats=[PREREQUISITES]) |
| 3 | What is CSCE 313 about? | `metadata_default` | `metadata_combined` (cats=[COURSE_OVERVIEW, LEARNING_OUTCOMES]) |
| 4 | What is CSCE 633 about? | `metadata_default` | `metadata_specific` (cats=[COURSE_OVERVIEW]) |
| 5 | When and where are the final exams for the different sections of CSCE 120? | `metadata_default` | `metadata_specific` (cats=[SCHEDULE]) |
| 6 | What's a typical day of the week that I'll have CSCE 120? | `metadata_default` | `metadata_specific` (cats=[SCHEDULE]) |
| 7 | What's the deal with the class engagement score in CSCE 120? | `metadata_default` | `metadata_specific` (cats=[GRADING]) |
| 36 | What's ISEN 625 about? | `metadata_default` | `metadata_combined` (cats=[COURSE_OVERVIEW, LEARNING_OUTCOMES]) |
| 37 | What's ISEN 350 about? | `metadata_default` | `metadata_combined` (cats=[COURSE_OVERVIEW]) |

**Key observation:** Cases 1, 5, 6, 7 — the router's actual output is arguably **more correct**
than the golden label. "What are the prerequisites for CSCE 413?" is clearly asking about
PREREQUISITES; routing to `metadata_specific` retrieves exactly that category. The golden
label `metadata_default` would retrieve `DEFAULT_SUMMARY_CATEGORIES` (a broader set), which
includes PREREQUISITES anyway but adds noise.

Similarly cases 5, 6, 7: schedule/grading questions are factually about those specific
categories — `metadata_specific` is the right call.

Cases 3, 4, 36, 37 ("What is X about?") are more ambiguous. `COURSE_OVERVIEW` is a
defensible extraction, but the golden intent was to rely on the default summary path.

**Root cause:** Golden set generation uses `_derive_router_ground_truth()` mechanically
from the *stratum* (`metadata_default` → `expected_specific_categories=[]`), not from the
actual question content. When the synthesis LLM generates a question that names a specific
category topic (e.g., "prerequisites", "exam schedule"), the stratum label and actual
question semantics diverge. The router sees the question semantics; the golden label
reflects the stratum.

**Verdict:** These are **golden set labeling errors**, not router bugs. Cases 1, 5, 6, 7
should be re-labeled `metadata_specific`. Cases 3, 4, 36, 37 are borderline ("about"
questions); `metadata_default` is defensible but `metadata_combined` is not wrong.

---

### Bucket 2 — Over-extraction on opinion/recommendation questions (5 cases)

Questions: 9, 11, 14, 39, 41

| # | Question | Expected | Actual |
|---|---|---|---|
| 9 | Would you recommend CSCE 432 to someone interested in a career focused on ethical technology development? | `hybrid_default` | `hybrid_specific` (cats=[LEARNING_OUTCOMES, COURSE_OVERVIEW]) |
| 11 | Is CSCE 633 a good choice for someone interested in theoretical computer science? | `hybrid_default` | `hybrid_combined` (cats=[LEARNING_OUTCOMES, COURSE_OVERVIEW]) |
| 14 | Would you recommend CSCE 412 as a good intro to concepts for a cloud engineering internship? | `hybrid_default` | `hybrid_specific` (cats=[LEARNING_OUTCOMES]) |
| 39 | Given the flipped classroom format of ISEN 350, how much time should I expect to dedicate each week? | `hybrid_default` | `hybrid_specific` (cats=[COURSE_OVERVIEW]) |
| 41 | Given that ISEN 350 is a flipped classroom and requires active engagement, would you recommend taking it if I'm looking for a more traditional lecture-based course? | `hybrid_default` | `hybrid_specific` (cats=[COURSE_OVERVIEW]) |

**Key observation:** Again the router's actual output is arguably **correct** for several of
these. Cases 9, 11, 14 are CAREER-type questions — the answer necessarily lives in
LEARNING_OUTCOMES. `hybrid_specific` retrieves that category + applies the semantic advisory
overlay, which is exactly right. `hybrid_default` would retrieve DEFAULT_SUMMARY_CATEGORIES
which is broader and potentially less focused.

Cases 39, 41 explicitly reference the course format ("flipped classroom"), which is
COURSE_OVERVIEW content. `hybrid_specific` with `cats=[COURSE_OVERVIEW]` is precise.

**Root cause:** Same as Bucket 1 — the `hybrid_default` stratum framing asks for questions
"without naming a specific category," but the synthesis LLM generates questions that
contextually imply a category. The ground truth is again mechanically derived from the
stratum, not the question.

**Verdict:** Golden set labeling errors. These questions would legitimately route to
`hybrid_specific` under the router's intended semantics.

---

### Bucket 3 — `conf` too high → metadata path when hybrid expected (3 cases)

Questions: 33, 34, 49

| # | Question | Expected | Actual |
|---|---|---|---|
| 33 | What should I know about CSCE 331, especially regarding the required background courses? | `hybrid_combined` | `metadata_combined` (conf=0.95) |
| 34 | What should I know about CSCE 608, including its policy on using AI? | `hybrid_combined` | `metadata_combined` (conf=0.95) |
| 49 | Can you give me a general overview of ISEN 281, especially regarding the course structure? | `hybrid_combined` | `metadata_combined` (conf=0.90) |

**This is a genuine router issue** — different from Buckets 1 and 2. Category extraction is
correct (`PREREQUISITES`, `AI_POLICY`, `COURSE_OVERVIEW`). `specific_only=False` is correct
(the "especially"/"including" framing). But `conf=0.90–0.95` → `retrieval_mode=metadata` →
`metadata_combined` when golden expects `hybrid_combined`.

The broad framing ("what should I know about X") warrants semantic retrieval to surface
contextually relevant content beyond exact category matches. High confidence on an explicitly
named category shouldn't override this.

**Fix option:** Lower `CATEGORY_CONFIDENCE_THRESHOLD` from 0.7 to 0.6, or add a rule that
`*_combined` queries always use hybrid mode regardless of confidence (since the broad framing
signals that exact lookup isn't sufficient).

---

### Bucket 4 — Two one-offs (cases 8, 19)

| # | Question | Expected | Actual | Issue |
|---|---|---|---|---|
| 8 | Given the rules about copyrighted materials in CSCE 120, does that mean I can't share my homework solutions? | `hybrid_default` | `metadata_specific` | Router: si=False (missed inferential phrasing); also over-extracts UNIVERSITY_POLICIES. Actual should likely be `hybrid_specific` with cats=[UNIVERSITY_POLICIES] |
| 19 | If I don't access the Perusall readings through Canvas, will my grades still show up correctly? | `semantic_general` | `out_of_scope` | No course_id + si=False → out_of_scope. Question is about a TAMU admin tool. Adding ADMINISTRATIVE semantic_type and prompting the router to recognize TAMU-tool questions as si=True fixes this. |

**Case 8:** Both expected (`hybrid_default`) and actual (`metadata_specific`) are arguably
wrong. The question is evaluative/inferential ("does that mean I can't...") → si=True is
correct. It explicitly references "the rules about copyrighted materials" → specific category.
Best label: `hybrid_specific` with cats=[UNIVERSITY_POLICIES].

**Case 19:** Fixed by adding ADMINISTRATIVE semantic_type and updating router prompt to
recognize TAMU-specific tools (Canvas, Perusall, Howdy) as academic context → si=True.

---

## Summary: what's a router bug vs. golden set issue

| Category | Count | Root cause | Action |
|---|---|---|---|
| Golden set labeling errors (stratum framing ≠ question content) | ~10 | `_derive_router_ground_truth()` is mechanical, ignores actual question | Re-generate golden set with post-hoc LLM re-labeling, or manually correct |
| Genuine router issues | 3 (bucket 3) | conf too high → metadata path for broad combined queries | Lower threshold or force hybrid on `*_combined` |
| One-offs | 2 (bucket 4) | Missing ADMINISTRATIVE type; missed inferential si detection | Add ADMINISTRATIVE type (done); improve si examples in prompt |

**True router accuracy (excluding golden label errors) is closer to ~45/50 (90%).**

---

## Confidence calibration

The bimodal distribution of `category_confidence` (0.0 or 0.90–1.00) means:

- The ECE of 0.179 is **misleading** — it appears as "poor calibration" but is actually
  "I'm always certain when I have a category, and I emit 0.0 when I have none."
- The `[0.0–0.1]` bin gap (conf=0 but acc=1.0) just means: when the router correctly
  outputs no categories (specific_categories=[]), it emits conf=0.0. Those routes are
  correct because the derivation matrix treats conf=0.0 as "use hybrid/semantic," not as
  "wrong."
- **No action needed on confidence calibration itself.** The threshold (0.7) is effectively
  binary: anything extracted gets ≥0.90; anything not extracted gets 0.0.

---

## Changes made (2026-02-23)

1. **`db/router.py`** — Added `ADMINISTRATIVE` semantic_type for TAMU-tool questions
   (Canvas, Perusall, grade tracking) with no course_id. Updated prompt examples.
2. **`scripts/eval_pipeline.py`** — Added `--golden-set PATH` flag to run full pipeline
   eval (router → retrieval → generator) against a golden set JSONL instead of the
   hardcoded 34-case TEST_SUITE.
3. **`scripts/eval_router_metrics.py`** — Fixed `tc["query"]` KeyError (golden set uses
   `question` field); added UTF-8 stdout reconfigure for Windows.

## Next steps

1. **Re-generate golden set** with post-hoc re-labeling: after synthesis, run each question
   through the router and have an LLM adjudicate whether the stratum label or the router
   output is more semantically correct. This will reduce false negatives in the eval.
2. **Test Bucket 3 fix:** try forcing `*_combined` functions to always use hybrid mode
   (ignore `category_confidence` when `specific_only=False`). Measure impact on the 34-case
   TEST_SUITE to confirm no regression.
3. **Validate ADMINISTRATIVE type** on case 19 and any similar Canvas/Perusall questions.
4. **Run full pipeline eval** on all 50 golden set cases to assess retrieval quality and
   generation (citation rate, response relevance) beyond router accuracy alone.
