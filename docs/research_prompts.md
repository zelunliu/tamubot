# Research Prompts for TamuBot RAG Pipeline

Use each prompt as a separate Gemini Deep Research session.

---

## PROMPT 10: Reducing Latency and Token Usage While Improving RAGAS Answer Relevancy

I'm running a production LangGraph RAG chatbot (TamuBot, Texas A&M) and want to reduce end-to-end latency and token consumption while improving RAGAS answer relevancy. Please research concrete, evidence-backed techniques and compare specific approaches for my stack.

---

### Goal

Cut total pipeline latency from ~12–15s to under 8s, reduce unnecessary token usage, and raise RAGAS answer relevancy from ~0.50–0.55 to 0.70+, without sacrificing faithfulness (currently 0.87–0.97).

---

### Current State (measured across 3 eval runs, Mar 2026, n=10 each)

**Latency breakdown (mean):**
| Component | Latency |
|---|---|
| Router node | 3,200–3,900 ms |
| Generator node | 2,800–4,100 ms |
| Retrieval + rerank | 420–580 ms |
| Total pipeline | 11,400–14,600 ms |

**Token usage (estimated):**
- Input: ~3,200 tokens/query (system prompt + XML context, 5 chunks)
- Output: 186–342 tokens/query (varies by function type)
- TAMU gateway forces `max_tokens=4096` on all calls — actual usage is <350

**Quality:**
- RAGAS faithfulness: 0.87–0.97 ✓
- RAGAS answer relevancy: 0.50–0.55 ✗
- Router accuracy: 100% across all runs

**Stack constraints:**
- Router: Gemini 2.5 Flash, `thinking_budget=512`, temp=0, JSON output
- Generator: Gemini 2.0 Flash via TAMU OpenAI-compatible gateway (SSE streaming, min `max_tokens=4096`)
- TAMU gateway does NOT expose token counts in SSE response
- Generator system prompt includes Chain-of-Verification: model writes a `<thinking>` block with a verbatim quote before answering; `<thinking>` is stripped before delivery to user
- 5 chunks retrieved/reranked per query (Voyage AI rerank-2)
- Router prompt: ~130 lines, extracts 8 structured fields as JSON

---

### Research Questions

**1. Router latency (3.2–3.9s for JSON extraction)**
- What is the latency impact of `thinking_budget=512` on Gemini 2.5 Flash for a simple structured extraction task? Does thinking meaningfully improve JSON accuracy vs. a non-thinking call?
- For a 130-line prompt extracting 8 fields, what techniques reduce latency without accuracy regression: prompt compression, few-shot removal, smaller model, or structured output schema enforcement?
- Is there a documented latency difference between Gemini 2.5 Flash (thinking) vs. Gemini 2.0 Flash (no thinking) for JSON classification tasks?

**2. Generator latency and token waste**
- Chain-of-Verification: does forcing a `<thinking>` block with a verbatim quote before answering measurably improve faithfulness in RAG systems? What does the literature say? Is the tradeoff worth ~1–2s of extra generation time?
- With TAMU gateway forcing `max_tokens=4096` but actual output being <350 tokens, is there a way to reduce latency without reducing `max_tokens`? (e.g., stop sequences, output length control in system prompt)
- What prompt techniques reduce answer length without reducing answer quality (reducing output tokens → lower generation latency)?

**3. RAGAS answer relevancy at 0.50–0.55**
- RAGAS AnswerRelevancy measures whether the answer addresses the question. At 0.50–0.55 with faithfulness at 0.87–0.97, what is the most likely cause: over-verbose answers, answers covering topics not asked about, or a metric measurement issue?
- What prompt changes are most effective for improving answer relevancy specifically: tighter focus instructions, explicit "only answer what was asked" constraints, or output format changes?
- Does reducing retrieved chunk count (5 → 3) improve relevancy by reducing off-topic context, or does it hurt faithfulness?

---

### Deliverables

1. **Compare 3 approaches** for each of the 3 problem areas above, with trade-offs on: latency impact, quality impact, implementation complexity, and risk
2. **A ~2-page implementation summary** suitable for pasting directly to an AI coding agent, covering:
   - Exact changes to make (file, parameter, prompt text)
   - Expected measurable impact per change
   - What NOT to change (faithfulness is working; don't break it)
   - Suggested order of implementation (highest ROI first)


---

## PROMPT 8: Conversation Memory State Management for LangGraph RAG

I have a production LangGraph-based RAG chatbot and I want to research the best, easiest-to-implement improvements to its conversation memory. Please do a deep research pass and give me concrete, practical recommendations.

---

### System Overview

The system is a **course-advising chatbot** (Texas A&M University) built on LangGraph with the following pipeline:

**Pipeline nodes (in order):**
router → history_inject → [conditional: out_of_scope | anchor → eval_search → retrieval → schedule_filter → merge | retrieval] → generator → history_update → END

**State contract (TypedDict):**
- `query`: raw user question
- `rewritten_query`: LLM-rewritten query (modified by history_inject_node)
- `function`: routing decision ("hybrid_course", "recurrent", "semantic_general", "out_of_scope")
- `course_ids`, `intent_type`, `specific_categories`: router-extracted metadata
- `retrieved_chunks`: list of document chunks from MongoDB+Voyage AI vector search
- `answer`: final generated answer string
- `answer_stream`: streaming iterator (NOT checkpointed — stripped before LangGraph checkpoint)
- `trace`: Langfuse observability object (NOT checkpointed — stripped before LangGraph checkpoint)
- `timing_ms`, `node_trace`, `error`: observability metadata
- `history`: list of ConversationMessage (role, content, router_result)
- `history_summary`: str (field exists, NEVER populated — reserved)
- `turn_number`: int
- `session_id`: str

**Checkpointing:**
- Uses LangGraph's MemorySaver (default) or SqliteSaver (`V4_CHECKPOINTER_BACKEND=sqlite`)
- Session ID from Streamlit → mapped to LangGraph `thread_id` via in-memory dict
- Non-serializable fields (answer_stream, trace) are stripped before each checkpoint via a wrapper

**History management (current):**
- `history_inject_node`: prepends the last 3 user/assistant turns (6 messages) as plain text to `rewritten_query`
- `history_update_node`: appends current turn, slides window to last `V4_MAX_HISTORY_TURNS=6` turns (12 messages), then clips — no summarization
- `history_summary` field exists in state but is never written to

**Router behavior:**
- Classifies query fresh each turn (does NOT read history)
- Outputs: course_ids, intent_type, specific_categories, recurrent_search, rewritten_query
- History is injected AFTER routing to preserve fresh per-turn classification

---

### What Is Already Working Well

- LangGraph checkpointing correctly persists history across turns
- Router runs on clean query (avoids context contamination of routing)
- history_inject runs after router and enriches rewritten_query with plain-text context window
- Non-serializable fields are handled (stripped/re-injected around checkpoints)
- The system is live and functional; we want improvements, not a rewrite

---

### Pain Points / What Is Missing

1. **History sliding window loses context** — after 6 turns, older context is permanently dropped. The `history_summary` field exists but is never populated.

2. **History injection is text-only** — the last 3 turns are prepended as a plain-text block to the query string. There is no structured message history passed to the generator LLM, only to the retrieval query.

3. **Router is history-blind** — it classifies purely on the current turn. Follow-up queries like "and what about the lab sections?" get classified as new intents rather than continuations, sometimes causing missed context.

4. **Generator doesn't see history messages natively** — the generator LLM receives `rewritten_query` (which may have 3-turn text prepended) + retrieved chunks. There is no explicit `messages: [...]` array with system/user/assistant turns.

5. **No "is follow-up" detection** — the router could tag turns as follow-ups vs new questions, but currently does not. This would change retrieval behavior (e.g., inherit previous course_ids on follow-up).

6. **answer_stream non-serializable problem** — the streaming iterator cannot be checkpointed, so the system collects all tokens into a list and replays. This works but is inelegant.

7. **Two parallel graph implementations** — `build_graph()` (stateless) and `build_graph_with_memory()` duplicate all edge definitions. Changes must be applied twice.

---

### Research Questions

Please answer all of these with a focus on **LangGraph-native solutions** and **minimal additional complexity**:

**1. Conversation history summarization**
- What is the standard LangGraph pattern for automatic history summarization? (e.g., using a summarization node that fires when history exceeds N turns)
- Should this run as a dedicated node after history_update, or inline within history_update_node?
- What LLM call pattern is recommended (separate summarization call vs. asking the generator to compress)?
- How should `history_summary` be injected into the pipeline — into `rewritten_query` alongside the window, or directly into the generator's system prompt?

**2. Structured message history vs. text injection**
- Should I pass a proper `messages: [{role, content}]` list to the generator LLM instead of prepending context to `rewritten_query`?
- What are the trade-offs between text-injection (current) vs. structured multi-turn messages for an OpenAI-compatible API?
- Can I do both: use structured history for the generator while keeping text injection for retrieval query enrichment?

**3. Follow-up detection and intent continuity**
- What is the simplest LangGraph pattern to detect follow-up queries (continuation of previous topic) vs. new questions?
- Should this be a flag the router emits, or a separate node?
- If a query is a follow-up, what state should be inherited from the previous turn (course_ids, specific_categories, intent_type)?

**4. LangGraph tools and MCP integration**
- I plan to add tool-use capabilities (web search, calendar lookup, etc.) and MCP servers in the future.
- How does adding tools/MCP affect conversation memory management? What changes to state design should I make now to avoid rework later?
- What LangGraph patterns (e.g., ToolNode, interrupt_before/after) interact with checkpointed conversation history?

**5. Simplifying two graph variants**
- Is there a clean pattern to unify `build_graph()` (stateless) and `build_graph_with_memory()` into a single graph that conditionally enables memory?
- Or is the dual-graph approach idiomatic in LangGraph?

**6. Session persistence**
- The session_id → thread_id mapping lives in an in-memory dict and is lost on restart. Should I persist this mapping (e.g., in the same SQLite DB as LangGraph's checkpointer)?
- What is the idiomatic LangGraph pattern for managing thread IDs across process restarts?

---

### Constraints and Preferences

- **LangGraph-native first**: prefer patterns that use LangGraph's built-in capabilities (checkpointing, interrupt, ToolNode) over external state stores
- **Minimal complexity**: easy wins over architecturally pure. Incremental improvements over rewrites.
- **No breaking changes to the pipeline node interface**: each node must remain `fn(state, registry) -> dict`
- **OpenAI-compatible API**: the system uses TAMU's OpenAI-compatible gateway (streaming only, min max_tokens=4096)
- **MongoDB + Voyage AI**: retrieval uses MongoDB Atlas vector search + Voyage AI embeddings; stateless per call

---

### Deliverables I Want

1. A **prioritized list** of conversation memory improvements, from easiest/highest-impact to most complex
2. For each improvement: a concrete description of what changes (which nodes, which state fields, which LangGraph APIs)
3. Any **LangGraph documentation links or official patterns** I should read
4. A note on **what NOT to do** (anti-patterns to avoid for this use case)

---

## PROMPT 7: RAG Evaluation Workflow — Golden Test Set, Stage Isolation, Experiment Tracking, and Hyperparameter Optimization

I'm building **TamuBot**, a production RAG chatbot for Texas A&M University students. I need a **complete, practical evaluation workflow** designed for my exact stack and resource constraints. The goal is to systematically optimize the pipeline — prompts, retrieval parameters, architecture choices — by measuring the right things cheaply at each stage. Give me state-of-the-art techniques grounded in evidence, not generic advice.

Do NOT recommend tools I would need to integrate from scratch. Every recommendation must be implementable with the components already in use (Langfuse, RAGAS, MongoDB Atlas, Voyage AI, Gemini Flash) or with a minimal, well-justified addition.

---

### Project Context

**Stack:** Python 3.12, Streamlit, MongoDB Atlas M0, Voyage AI (voyage-3 + rerank-2), Gemini 2.5 Flash (router) + Gemini 2.0 Flash (generator), google-genai raw SDK, Langfuse (custom REST client, 5-span hierarchy), RAGAS (Faithfulness + AnswerRelevancy, async background thread, Voyage AI as critic embeddings).

**3-stage pipeline:**

```
[Stage 1 — Router]  db/router.py
    Gemini 2.5 Flash, temp=0, thinking_budget=512
    Extracts: course_ids, specific_categories, category_confidence (0–1),
              specific_only, semantic_intent, semantic_type, rewritten_query
    Derives in pure Python: function (8 types), retrieval_mode (metadata/hybrid/semantic)
    8 functions: metadata_default/specific/combined,
                 hybrid_default/specific/combined, semantic_general, out_of_scope

[Stage 2 — Retrieval + Rerank]  db/search.py + db/reranker.py
    Metadata path: search_by_course_categories() — exact index lookup, no embedding
    Hybrid path:   hybrid_search() — $vectorSearch + BM25 → manual RRF → Voyage rerank-2
    Semantic path: search_semantic() — pure vector, no course filter
    Post-retrieval: _deduplicate_chunks() — keep best chunk per (course_id, category)
    Config: FUNCTION_RETRIEVAL_CONFIG{function: {retrieve_k, rerank_k}}

[Stage 3 — Generator]  db/generator.py
    Gemini 2.0 Flash, XML-tagged context, function-adaptive system prompts,
    semantic_type advisory overlay, [Source N] citations, recency-bias ordering
```

**Current eval harness** (`scripts/eval_pipeline.py`):
- 34 `TestCase` dataclass instances, all targeting CSCE 638 + CSCE 670 (Spring 2026)
- Captures per-run: `function_correct`, `course_ids_correct`, `specific_categories_correct`, `chunks_retrieved`, `has_citations`, `citation_count`, `latency_*_ms`, `response_preview`
- Outputs: JSONL (machine-readable) + Markdown report (human/LLM-readable) per run
- 34/34 function accuracy on dry-run (router only)
- Full pipeline eval not yet run against real MongoDB data

**Current RAGAS** (`db/observability.py`):
- `compute_ragas_metrics(question, contexts, answer, trace_id)` — Faithfulness + AnswerRelevancy
- Uses Gemini 2.0 Flash as critic LLM, Voyage AI voyage-3 as critic embeddings
- Called via `run_ragas_background()` — fire-and-forget daemon thread per live query
- Scores uploaded to Langfuse via `/api/public/scores`
- NOT integrated with eval_pipeline.py — only runs on live production traffic

**Key constraints:**
- Free tiers only: Gemini Flash (15 RPM free tier), Voyage AI (limited monthly tokens), MongoDB M0
- No LangChain/LlamaIndex — raw SDK calls throughout
- Langfuse Datasets API available — not yet used

---

### What I Need Researched

#### Part A — Golden Test Set Construction (~50 Questions)

I need to build a "golden" evaluation dataset of ~50 questions with known correct answers, stratified across all 8 function types. The data source is real: ~3,100 chunks already in MongoDB (259 course sections, 11 categories each). Research the following:

**1. Generation strategy: question + reference answer from real MongoDB chunks**

The most honest method is to generate questions FROM the actual chunk content, so the reference answer is verifiable against the source. Research:
- What is the proven "reverse QA generation" technique for RAG evaluation? (e.g., pick a chunk, ask LLM "given this text, write a question a student would ask") — cite the RAGAS documentation and any papers validating this approach.
- For **metadata function types** (metadata_default/specific/combined): these map to exact category lookups. Is a "reference answer" even necessary for these, or is it sufficient to verify that the retrieved chunk contains the correct category and the citation appears in the response? What does the literature say about evaluating factual-lookup RAG without a full reference answer?
- For **hybrid function types** (hybrid_default/specific/combined): these involve subjective/advisory answers where the "correct" answer is grounded in specific chunks but not verbatim quotable. What is the standard approach for generating reference answers for opinion/evaluation questions in domain-specific RAG?
- For **semantic_general**: no course ID, full-corpus retrieval. How do you construct a reference for "which courses help with an ML career?" when the answer depends on what's in the corpus? What proxy is used?
- For **out_of_scope**: only need to verify the canned response is returned. No reference answer needed.

**2. Stratification: how many cases per function type?**

Current 34 cases are distributed unevenly (7 metadata_specific, 3 metadata_default, 2 metadata_combined, etc.). For a golden 50-case set covering 8 function types, research:
- What is the minimum cases-per-function-type for statistical power to detect a meaningful regression (e.g., function accuracy drops from 100% to 85%)? Use a binomial proportion confidence interval calculation.
- For function types with fewer natural queries (hybrid_combined, metadata_combined — genuinely rare), is it better to have fewer high-quality cases or more synthetic ones?
- Should multi-course cases (2 courses) be treated as a separate stratum? The current test suite has 5 multi-course cases spread across function types.

**3. Coverage of the hyperparameter space**

The golden set should stress-test the `category_confidence_threshold=0.7` boundary — queries that hover between metadata and hybrid retrieval paths. How do you deliberately construct queries at this boundary? Should some golden cases be designed to fail under a wrong threshold setting, so you can measure threshold sensitivity?

**4. Human validation budget**

At ~50 questions generated by LLM from real chunks, what percentage requires human review before the set is reliable? Cite any studies on LLM-generated QA quality for domain-specific corpora (academic/technical text). What specific error types appear most often (hallucinated question, question answerable without the chunk, question too vague)?

---

#### Part B — Stage-Isolated Evaluation

The key challenge: I want to evaluate each stage independently, so a retrieval bug doesn't confound generator scores and vice versa. Research the proven techniques for this in multi-stage RAG pipelines.

**Stage 1 — Router evaluation (already partially implemented)**

The router extracts 7 variables + derives 2 more. Current `eval_pipeline.py` checks `function_correct`, `course_ids_correct`, `specific_categories_correct`. What is missing:
- **`category_confidence` calibration**: is the confidence score well-calibrated? (i.e., does confidence=0.9 actually correspond to ~90% accuracy on category extraction?) What is the standard calibration check — ECE (Expected Calibration Error) or reliability diagram — applied to a 50-case set?
- **`semantic_intent` precision/recall**: currently only checking bool correct. What edge cases should be in the golden set specifically to stress-test the semantic_intent=True boundary (evaluative language vs. factual language)?
- **`rewritten_query` quality**: the router rewrites queries for retrieval. How do you measure whether the rewrite actually improves downstream retrieval without running full retrieval? Is there a fast proxy (embedding similarity between rewrite and relevant chunk vs. original query and same chunk)?

Stage 1 evaluation requires NO LLM judge — all deterministic checks on `RouterResult` fields.

**Stage 2 — Retrieval evaluation (the hardest stage to evaluate cheaply)**

To evaluate retrieval in isolation, you need to know which chunks are "correct" for each golden query. Research:
- **Relevance labels without human annotation**: Can the reference answer from Part A be used to automatically label chunks as relevant (embedding similarity > threshold between reference answer and chunk content)? What threshold is appropriate for Voyage AI voyage-3 embeddings? Cite any benchmarks that validate this approach.
- **Recall@k**: for each golden query, what fraction of the labeled-relevant chunks appear in the top-k retrieved results? How do you compute this for the metadata path (where retrieval is exact) vs. hybrid path (where it's probabilistic)?
- **Reranker isolation**: the pipeline retrieves retrieve_k=10–30 then reranks to rerank_k=3–5. How do you measure whether the reranker is actually improving result quality vs. the pre-rerank ranking? Standard metric: NDCG improvement pre-rerank vs. post-rerank. What is the implementation for this given the Voyage rerank-2 API returns `relevance_score` per result?
- **RRF k_param sensitivity**: the manual RRF uses k_param=60. How do you measure the impact of varying this parameter on retrieval quality in your specific corpus? What range should be searched?
- **Metadata path shortcut**: `search_by_course_categories()` returns exact chunks without scoring. For the metadata path, retrieval quality = (correct category fetched AND correct course_id matched). This is fully determined by the router's `function_correct` and `specific_categories_correct`. Is additional retrieval eval needed for the metadata path?

Stage 2 evaluation requires NO LLM judge if relevance labels are derived from embedding similarity to reference answers. One Voyage embed call per golden case to create the reference embedding.

**Stage 3 — Generator evaluation (where LLM judge is justified)**

For the generator, research which metrics require an LLM judge and which can be computed cheaply:

- **Citation coverage** (FREE — already implemented): `citation_count > 0` and `has_citations` are already tracked in `eval_pipeline.py`. What additional citation-based metrics are valuable? Citation precision (do cited sources actually support the claim)? This requires LLM judge.
- **Faithfulness** (RAGAS, LLM required): the current implementation uses Gemini 2.0 Flash as critic. Research: does faithfulness vary significantly by function type? (metadata functions at temp=0 should have near-perfect faithfulness; hybrid functions at temp=0.1 are riskier.) Should faithfulness be measured separately per function type in the golden set?
- **Answer completeness for metadata functions** (CHEAP — deterministic): for `metadata_specific` queries (e.g., "What is the grading breakdown for CSCE 638?"), completeness = does the response contain the key fields from the source chunk (letter grades, percentages, etc.)? Can this be measured with regex or keyword matching against the reference chunk, avoiding LLM entirely?
- **LLM-as-judge cost minimization**: the current RAGAS runs Gemini 2.0 Flash as critic per query (expensive). Research the "tiered evaluation" pattern:
  1. First run cheap deterministic checks (citation count, keyword overlap, function accuracy)
  2. Run LLM judge only on cases that pass tier 1 OR on a stratified sample
  3. Use embedding similarity as a fast proxy for answer relevancy before committing to LLM judge
  What correlation exists between embedding-based answer similarity and RAGAS AnswerRelevancy scores? Cite any RAGAS documentation or benchmarks on proxy metric reliability.
- **Judge model bias**: current setup uses Gemini 2.0 Flash as both generator AND judge. Research: what is the measured self-evaluation bias for same-family LLM judges vs. cross-family (e.g., using Gemini to judge its own output)? Should the judge be a different model (e.g., use a smaller Gemini model or a different provider) for the golden set eval?

---

#### Part C — Experiment Tracking with Langfuse Datasets

Langfuse has a Datasets API that allows batch experiments to be tracked and compared. I have not used this feature yet. Research:

**1. Langfuse Datasets workflow for RAG eval**
- `POST /api/public/datasets` — create a named dataset (e.g., "golden-v1")
- `POST /api/public/dataset-items` — add items: each item has `input` (the query), `expectedOutput` (reference answer), and `metadata` (function_expected, course_ids, etc.)
- Running the eval: for each dataset item, run the full pipeline, then `POST /api/public/dataset-run-items` to link the trace to the dataset item
- Comparing runs: in the Langfuse UI, compare `scores` across experiment runs on the same dataset
- What is the exact REST API call sequence to implement this against the `MinimalLangfuseClient` class in `db/observability.py`? Provide the specific endpoint names and request bodies.

**2. Experiment metadata tagging**
- Each eval run should be tagged with the hyperparameters being tested (e.g., `{"category_confidence_threshold": 0.7, "hybrid_retrieve_k": 12, "thinking_budget": 512}`)
- How do you attach experiment metadata to a Langfuse dataset run so it's queryable in the UI?
- What naming convention for dataset runs enables easy comparison? (e.g., `baseline-2026-02-23`, `threshold-0.6-2026-02-24`)

**3. Score aggregation and comparison**
- Langfuse shows per-trace scores. For comparing two experiment runs on a 50-case dataset, how do you compute aggregate statistics (mean, p-value, confidence interval) from Langfuse score data?
- Is there a Langfuse export API to pull all scores for a dataset run into a pandas DataFrame for statistical analysis?
- What is the recommended statistical test for comparing two RAG pipeline configurations on a binary metric (function accuracy: 0 or 1) with N=50 samples? McNemar's test vs. bootstrap? For a continuous metric (RAGAS faithfulness: 0–1), what test applies?

---

#### Part D — Hyperparameter Optimization Workflow

The goal is to systematically optimize the pipeline using the golden test set. Research which parameters have the most leverage and how to search them efficiently with limited compute.

**Parameters to optimize (ranked by expected impact):**

| Parameter | Current value | Where it lives | Stage affected |
|---|---|---|---|
| `CATEGORY_CONFIDENCE_THRESHOLD` | 0.7 | config.py | Router → retrieval path selection |
| `hybrid_*` retrieve_k | 10–15 | FUNCTION_RETRIEVAL_CONFIG | Retrieval recall |
| `hybrid_*` rerank_k | 3–4 | FUNCTION_RETRIEVAL_CONFIG | Context window size |
| `thinking_budget` | 512 | router.py | Router accuracy vs. cost |
| per-function temperature | 0.0/0.1 | generator.py | Generator faithfulness |
| RRF k_param | 60 | search.py `_rrf_fuse()` | Fusion balance |
| `DEFAULT_SUMMARY_CATEGORIES` | 3 categories | config.py | metadata_default coverage |

Research:
1. **Search strategy for small eval sets**: with N=50 test cases and ~7 hyperparameters, what search strategy is appropriate? Grid search is only feasible for 1-2 parameters at a time. Research whether Bayesian optimization (e.g., Optuna) is justified at this scale, or whether a structured one-parameter-at-a-time sweep is more reliable given the high variance of LLM outputs.
2. **The `category_confidence_threshold` is the highest-leverage parameter**: it controls the metadata vs. hybrid retrieval path. Values to test: 0.5, 0.6, 0.7, 0.8, 0.9. What metric directly captures whether this threshold is set correctly? (Hint: it's the interaction between `function_correct` and retrieval recall@k — if threshold is too high, metadata path fires on uncertain categories and retrieval misses.) How do you measure this interaction in a single experiment run?
3. **Temperature sensitivity for hybrid functions**: `hybrid_*` functions use temp=0.1 for the generator. Research: what is the measured impact of temperature on RAGAS Faithfulness for grounded-generation tasks? Is there published evidence that temp=0.0 reduces hallucination rates for retrieval-augmented generation? Should temp be part of the golden set optimization, or is the effect too small to measure with N=50?
4. **Avoiding overfitting to the golden set**: with a 50-case eval set, there's a risk of optimizing hyperparameters to the specific test cases. Research the standard practice for RAG hyperparameter tuning: should a held-out validation split be maintained (e.g., 40 optimization + 10 held-out)? What contamination risks exist when the golden set is generated from the same corpus that retrieval indexes?
5. **Cost of a full experiment run**: each full pipeline run on 50 cases makes: 50 router LLM calls (Gemini 2.5 Flash) + up to 50 embedding calls (Voyage voyage-3) + up to 50 rerank calls + up to 50 generator calls (Gemini 2.0 Flash) + up to 50 RAGAS judge calls (Gemini 2.0 Flash). At free tier limits (15 RPM Gemini, Voyage monthly limits), how many full experiments can run per day? What's the minimum viable experiment that tests only the parameters of interest without running the full pipeline (e.g., router-only sweep vs. retrieval-only sweep vs. generator-only sweep)?

---

#### Part E — Metrics, Statistics, and Trend Tracking

**1. Primary metrics per stage (what to track in every experiment run)**

| Stage | Metric | Computation | Requires LLM? |
|---|---|---|---|
| Router | function_accuracy | exact match | No |
| Router | course_id_accuracy | set match | No |
| Router | category_accuracy | subset match | No |
| Router | category_confidence_calibration | ECE | No |
| Retrieval | recall@k | labeled relevant chunks in top-k | No (embedding similarity label) |
| Retrieval | NDCG improvement | pre- vs. post-rerank | No |
| Generator | citation_coverage | `[Source N]` regex | No |
| Generator | faithfulness | RAGAS | Yes (LLM) |
| Generator | answer_relevancy | RAGAS | Yes (LLM + embed) |
| End-to-end | latency p50/p95 per function | from `latency_*_ms` fields | No |

Research: is this metric set sufficient to detect the failure modes that matter most? What important failure mode would go unmeasured by this set?

**2. Trend detection across runs**

Each eval run produces a JSONL file with per-case metrics. Research:
- What is the minimal Python implementation (no external ML frameworks) for detecting a statistically significant regression between two experiment runs? Specifically: if function_accuracy drops from 34/34 to 32/34, is that a real regression or noise? What N is needed to distinguish signal from noise at 95% confidence?
- For continuous metrics (RAGAS faithfulness 0–1), what N is needed to detect a meaningful improvement (e.g., 0.75 → 0.85) with 80% power using a paired t-test or Wilcoxon signed-rank test?
- Should the golden test set be "frozen" (fixed questions, fixed reference answers) across all experiment runs, or should it be refreshed when the underlying corpus changes significantly (e.g., after ingesting all 7,889 sections)?

**3. Langfuse dashboard design**
- Which Langfuse score names should be standardized across all experiment runs so they're comparable in the dashboard? Recommend a naming convention for scores from: eval_pipeline.py (deterministic), RAGAS (LLM judge), and any future custom metrics.
- How do you use Langfuse "tags" to group traces from a single eval run so you can filter and compare in the UI?

---

### Output Format

1. **Golden test set construction recipe** — step-by-step process: how to sample MongoDB chunks, generate questions, generate reference answers (with concrete LLM prompts), human validation checklist, final distribution across 8 function types

2. **Stage evaluation implementation plan** — for each stage: exact metric, how it's computed, whether it needs LLM, estimated cost per 50-case run, and which existing `EvalResult` fields already capture it vs. what needs to be added to `eval_pipeline.py`

3. **Tiered LLM-as-judge strategy** — decision tree: when to skip LLM judge, when to use cheap proxy, when to run full RAGAS, with estimated cost savings vs. information loss

4. **Langfuse Datasets integration code** — concrete REST API calls (compatible with `MinimalLangfuseClient`) to create dataset, add items, and link experiment run traces

5. **Hyperparameter sweep plan** — which parameters to sweep first, what grid, how to read the results, estimated total experiment cost (API calls) for a complete sweep

6. **Statistics cheat sheet** — for N=50: minimum detectable effect size for function accuracy (binomial) and RAGAS scores (continuous), with Python one-liners using `scipy.stats`

7. **References** — cite specific papers or documentation for: reverse QA generation, retrieval recall@k for RAG eval, LLM-as-judge bias, calibration of confidence scores, RAG hyperparameter optimization

---

## PROMPT 1: Chunking & Preprocessing Pipeline

(COMPLETED — we chose Gemini 2.5 Flash direct PDF parsing. See test_gemini_parse.py for implementation.)

**Summary of decisions made:**
- Gemini 2.5 Flash reads PDFs natively (multimodal) and outputs structured JSON in a single API call
- 11 semantic categories: COURSE_OVERVIEW, INSTRUCTOR, PREREQUISITES, LEARNING_OUTCOMES, MATERIALS, GRADING, SCHEDULE, ATTENDANCE_AND_MAKEUP, AI_POLICY, UNIVERSITY_POLICIES, SUPPORT_SERVICES
- Boilerplate university policies: only policy NAMES are extracted per course; full text stored once as a "golden copy"
- Completeness checking: Gemini flags missing sections and data quality warnings per syllabus
- Tables preserved as Markdown format
- 1 API call per PDF, ~20-60 seconds each, free tier
- Tested on CSCE 221, CSCE 331, CSCE 481, CSCE 482, ISEN 210, ISEN 302

---

## PROMPT 2: Database, Embedding & Retrieval

(COMPLETED — we chose MongoDB Atlas + Voyage AI. See Prompt 4 results for details.)

---

## PROMPT 3: LLM Orchestration Architecture for RAG

I'm building a RAG chatbot (TamuBot) for Texas A&M students. I need to design the **full LLM orchestration layer** — every LLM call in the pipeline, from user input to final response. I want a fundamental, unbiased exploration of best practices for my specific use case before committing to an architecture.

### System context

- **Frontend**: Streamlit chat UI
- **LLM**: Gemini 2.5 Flash (fast, cheap, structured JSON output)
- **Database**: MongoDB Atlas with 3 collections:
  - `chunks` (~3K docs): denormalized syllabus chunks, each with `course_id`, `section`, `term`, `instructor_name`, `category` (one of 11 types: GRADING, SCHEDULE, AI_POLICY, PREREQUISITES, LEARNING_OUTCOMES, etc.), `content`, `embedding` (Voyage AI voyage-3, 1024 dims)
  - `policies`: deduplicated university boilerplate policies (Academic Integrity, ADA, FERPA, etc.)
  - `courses`: one doc per section with full metadata, for aggregate queries
- **Search functions available**: `hybrid_search` (RRF fusion of vector + BM25), `search_semantic` (pure vector), `search_by_course` (metadata filter), `get_policy` (policy lookup), `aggregate_query` (count/comparison)
- **Scale**: 259 course sections now (CSCE + ISEN), expanding to all departments (~7,889 sections)

### What I need researched

**Part A — How many LLM calls do we actually need?**

Don't assume it's exactly two. Explore the full spectrum:
1. Single LLM call (stuff context + generate)
2. Two calls: router/classifier → retrieval → generator/synthesizer
3. Three+ calls: router → retrieval → re-ranker/filter → generator
4. Agentic loops (LangGraph, Google ADK, tool-calling agents)

For each pattern: when is it appropriate, what are the latency/cost/accuracy tradeoffs, and at what scale does each make sense? What's the right choice for ~3K chunks growing to ~50K?

**Part B — Inlet LLM (pre-retrieval)**

Whatever sits between the user's raw query and the database:
1. **Query classification**: How to distinguish queries that need retrieval (e.g. "What's the grading policy for CSCE 120?") from conversational queries that don't (e.g. "Howdy", "Thanks", "What can you help me with?")?
2. **Intent taxonomy**: What intents should we support? Consider: single-course lookup, cross-course comparison, aggregation/counting, policy lookup, instructor queries, schedule queries, conversational/chitchat, out-of-scope. Is this over-engineered or missing things?
3. **Entity extraction**: Pulling out course_id, section, category, instructor name, term from natural language. How to handle ambiguity ("the 400-level AI courses")?
4. **Retrieval strategy mapping**: How does each intent map to which search function? Should the router output a specific search function name, or just intent + entities?
5. **Query rewriting/expansion**: Should the inlet LLM rewrite vague queries before embedding? (e.g. "late work" → "late work policy attendance makeup")
6. **Safety/guardrails**: Handling off-topic queries, prompt injection attempts, and queries outside our knowledge domain gracefully
7. **Confidence and fallback**: When the router isn't sure, what's the fallback? Always run vector search? Ask the user to clarify?

**Part C — Outlet LLM (post-retrieval)**

Whatever takes retrieved documents and produces the final answer:
1. **Context formatting**: How to structure retrieved chunks in the prompt? Just concatenate? Include metadata headers? Numbered sources?
2. **Context window management**: With 5-10 retrieved chunks, how to fit them efficiently? Should we pre-filter/truncate before passing to the LLM?
3. **Grounding and hallucination prevention**: Best prompt patterns to keep the LLM strictly grounded in retrieved context. How to handle contradictions across different sections' syllabi.
4. **Citation/attribution**: How to make the LLM cite which course/section/category each fact came from, so the user can verify. Inline citations vs. footnotes vs. source list.
5. **No-results handling**: When retrieval returns nothing or low-relevance results, how should the outlet LLM respond? Admit it doesn't know vs. suggest related queries?
6. **Should it receive the router's classification?** Does passing the intent/entities to the generator help it produce better answers?
7. **Response format**: Should the outlet LLM be told the user's likely goal (e.g. "the user wants to compare grading policies") to structure its response appropriately?

**Part D — Practical considerations**

1. **Latency budget**: Users expect <3s for a chatbot response. If Gemini Flash is ~0.5-1s per call, how many LLM calls can we afford?
2. **Cost at scale**: With ~100 daily active users, what's the cost profile for each architecture?
3. **Caching**: Which LLM calls are cacheable? (e.g. identical course lookups)
4. **Streaming**: Can we stream the final response while still using a router? How does that work architecturally?
5. **Evaluation**: How to measure if our orchestration is actually working well? What metrics matter (retrieval precision, answer faithfulness, latency)?

### Example queries to consider in your analysis

- "Howdy!" (conversational — no retrieval needed)
- "What's the grading policy for CSCE 120?" (single course, specific category)
- "Who teaches CSCE 120 section 500?" (direct metadata lookup)
- "Compare AI policies across all 400-level CSCE courses" (cross-course comparison, filtered)
- "How many sections of CSCE 120 are there?" (aggregation)
- "Can I use ChatGPT for homework in CSCE 120?" (specific policy within a course)
- "What are the prerequisites for CSCE 120?" (single course, specific category)
- "Which CSCE courses have open-book finals?" (cross-course scan)
- "Tell me about the late work policy" (ambiguous — which course? or university-wide?)
- "What's the meaning of life?" (out of scope, gracefully reject)

### Output format

Give me:
1. **Recommended architecture** with justification — how many LLM calls, what each does, why
2. **Detailed prompt templates** for each LLM call (I'll adapt these into code)
3. **Intent taxonomy** with mapping to retrieval strategies
4. **Edge case handling** — how each example query flows through the system
5. **Tradeoff analysis** — what we gain/lose with this architecture vs. simpler/more complex alternatives
6. **2-3 reference implementations** (GitHub repos, blog posts, papers) that follow similar patterns
7. **Implementation priority** — what to build first, what can be added later

---

## PROMPT 4: MongoDB Atlas Integration

(COMPLETED — research done. Summary below.)

**Decisions made:**
- MongoDB Atlas M0 (free tier), 3 collections: chunks (denormalized), policies (deduplicated), courses (metadata)
- Embedding: Voyage AI voyage-3 (1024 dims), with contextual anchors prepended before embedding
- Hybrid search: $vectorSearch + $search with $rankFusion (RRF), MongoDB 8.0
- Ingestion: idempotent upserts, SHA-256 policy deduplication, batch embeddings (50/batch)
- Indexes: vector_index (vectorSearch), text_index (search), compound metadata index
- Scale: works on M0 for 259 sections, scales to 7,889+ without re-architecture

**Key references:**
- https://github.com/mongodb/chatbot
- https://github.com/mongodb-industry-solutions/manufacturing-car-manual-RAG
- https://github.com/mongodb-developer/Google-Cloud-RAG-Langchain
- https://github.com/mongodb-industry-solutions/document-intelligence

---

## PROMPT 5: Free Evaluation Framework for a 3-Stage RAG Pipeline

I'm building **TamuBot**, a RAG chatbot for Texas A&M University students. I need to choose and implement a **free, open-source evaluation and observability stack** that can measure data flow, per-stage latency, token costs, and answer quality across my 3-stage pipeline. I do NOT use LangChain or LlamaIndex — the pipeline is implemented with raw API SDK calls (google-genai, voyageai, pymongo). The evaluation framework must work with custom code.

### System Architecture

**Stack:** Python 3.12 + Streamlit, Router LLM: Gemini 2.5 Flash, Generator LLM: Gemini 2.0 Flash, Embeddings+Reranker: Voyage AI (voyage-3 + rerank-2), Database: MongoDB Atlas M0 (chunks, policies, courses collections), Search: manual RRF fusion of $vectorSearch + BM25 $search.

**3-Stage Pipeline:**

```
User query
    │
[Stage 1 — Router]  db/router.py
    Gemini 2.5 Flash (temp=0, max_output=512, thinking_budget=512)
    Output: {intent, course_ids, category, policy_name, rewritten_query, confidence}
    8 intents: single_course_lookup, multi_course_comparison, aggregation_query,
               policy_lookup, schedule_query, instructor_query, general_academic, out_of_scope
    │
[Stage 2 — Retrieval + Rerank]  db/search.py + db/reranker.py
    A) Voyage AI embed(rewritten_query) → 1024-dim vector
    B) $vectorSearch: top-20 candidates
    C) $search BM25: top-20 candidates
    D) Manual RRF fusion → top-20
    E) Voyage rerank-2 cross-encoder → top-5 final chunks
    (multi_course_comparison: runs A-E per course in parallel, then balanced interleave)
    │
[Stage 3 — Generator]  db/generator.py
    Gemini 2.0 Flash (temp=0.2, max_output=4096)
    Input: XML-tagged context chunks + intent-adaptive system prompt + user question
    Output: grounded answer with [Source N] inline citations
```

**Intent → retrieval mapping (current, global k=20→5 for all intents):**

| Intent | Retrieval path | Filter |
|---|---|---|
| single_course_lookup | hybrid_search | course_id + category |
| multi_course_comparison | multi_course_retrieve (parallel per course) | course_id per course |
| aggregation_query | aggregate_query() on courses collection | course_id, category |
| policy_lookup | get_policy() regex match | policy_name |
| schedule_query | hybrid_search | course_id + SCHEDULE category |
| instructor_query | hybrid_search | course_id + INSTRUCTOR category |
| general_academic | hybrid_search | course_id if present, else none |
| out_of_scope | none | — |

### Current Struggles

1. **No per-stage observability** — no latency breakdown, no token counts per stage, no retrieval quality signal.
2. **No answer quality measurement** — no ground truth dataset, no automated faithfulness or relevance scoring.
3. **Multi-course comparison whitespace bug** — near-duplicate chunks from multiple sections of the same course cause Gemini to pad markdown table cells with thousands of spaces, wasting the token budget.
4. **Global k values** — RETRIEVAL_TOP_K=20 and RERANK_TOP_K=5 are the same for all intents even though a schedule_query needs 1 chunk and a comparison needs many.

### What I Need

**Part A — Framework Landscape (free/open-source only)**

Survey and compare for our specific stack (no LangChain, raw SDK calls):
1. **RAGAS** — does it require a reference dataset? What metrics without ground truth (faithfulness, answer relevancy, context precision)?
2. **TruLens** — RAG Triad (groundedness, context relevance, answer relevance). Integration with custom non-framework code?
3. **Arize Phoenix** — LLM tracing, span-level metrics. Is self-hosted version truly free? How to instrument raw SDK calls?
4. **Langfuse** — open-source, self-hostable. How to capture token counts from google-genai `usage_metadata`?
5. **MLflow** — can `mlflow.tracing` work for RAG pipelines without LangChain?
6. **DeepEval** — RAG metrics via LLM-as-judge, no ground truth required.
7. **W&B Weave** — tracing and eval. Free tier limits?
8. **OpenTelemetry** — for pure latency measurement without LLM-specific tooling.

For each: integration complexity with raw SDK code, what it can/cannot measure, true free tier limits at ~100 DAU.

**Part B — Metrics That Matter Per Stage**

*Stage 1 (Router):* intent classification accuracy, entity extraction accuracy, latency p50/p95, token input/output counts, confidence score calibration.

*Stage 2 (Retrieval + Rerank):* embedding latency, MongoDB query latency (vector vs. BM25 vs. fused), reranker latency and score distribution, retrieval precision@k without labels, near-duplicate chunk rate.

*Stage 3 (Generator):* generation latency, token input/output (context XML is 2000-5000 tokens), faithfulness score, answer relevance score, citation coverage, response length by intent.

*End-to-end:* latency breakdown (router % / retrieval % / generation %), error rate per intent, p90 total latency target: <5 seconds.

**Part C — Implementation for Our Stack**

How do we instrument without LangChain? Specifically:
- Capturing Gemini token counts from `response.usage_metadata` (prompt_token_count, candidates_token_count)
- Adding span-level tracing to our `route_retrieve_rerank()` and `generate()` functions
- Integrating with our existing `scripts/eval_pipeline.py` (33 test cases, logs to JSONL)
- Minimum code changes to add observability — we don't want to refactor the pipeline

**Part D — LLM-as-Judge Without Ground Truth**

We have no gold-standard Q&A dataset. Research:
1. What prompts score faithfulness, relevance, completeness for a domain-specific RAG system?
2. Should we use Gemini 2.5 Flash as judge or a different model to avoid self-judging bias?
3. How many test cases give statistically meaningful eval results? Our current 33 cases — enough?
4. What is the minimum viable ground truth dataset to build, and how?

**Part E — Token Cost Model**

Estimate monthly cost at 100 DAU, 10 queries/user/day:
- Gemini 2.5 Flash (router): ~512 token input + ~100 output per query
- Gemini 2.0 Flash (generator): ~3000-5000 token input + ~500-1000 output per query
- Voyage AI voyage-3 embeddings: 1 embedding per query
- Voyage AI rerank-2: 20 docs × ~200 tokens each per query
- What free tiers are available for each service and when do we exceed them?

### Output Format

1. **Framework recommendation** — which single tool (or two-tool combo) for our stack, with justification
2. **Integration recipe** — concrete Python code wrapping our `route_retrieve_rerank()` and `generate()` to emit traces
3. **Metrics priority list** — ranked by implementation effort vs. insight value
4. **LLM-as-judge prompt templates** for faithfulness, relevance, context precision
5. **Token cost table** at 10/100/1000 DAU
6. **Minimum viable eval set** — how many examples, what structure
7. **Reference implementations** — GitHub repos doing RAG eval without LangChain

---

## PROMPT 6: System Optimization — Per-Intent Retrieval Tuning, Prompt Engineering, and Architecture

I'm building **TamuBot**, a RAG chatbot for Texas A&M University students. I need evidence-based recommendations for optimizing my system across 8 intent types: how many chunks to retrieve, how many to keep after reranking, how to fix the multi-course comparison bug, and how to improve both the router and generator prompts.

### Current System State

**Pipeline:** Router (Gemini 2.5 Flash) → Retrieval+Rerank (MongoDB hybrid + Voyage rerank-2) → Generator (Gemini 2.0 Flash). Global settings: RETRIEVAL_TOP_K=20, RERANK_TOP_K=5 (same for all intents).

**Data:** 259 course sections, Spring 2026 (CSCE + ISEN departments). Each syllabus → 8-14 chunks across 11 semantic categories. University boilerplate policies stored separately (one golden copy each). Expanding to ~7,889 sections when all departments are ingested.

**Router accuracy from eval (33 test cases, dry-run):**

| Intent | Accuracy | Notes |
|---|---|---|
| single_course_lookup | 5/5 (100%) | |
| multi_course_comparison | 4/4 (100%) | |
| aggregation_query | 4/4 (100%) | |
| policy_lookup | 4/4 (100%) | |
| schedule_query | 3/4 (75%) | "When is the final exam?" → misrouted to single_course_lookup |
| instructor_query | 4/4 (100%) | |
| general_academic | 1/4 (25%) | 3/4 queries with a course ID route to single_course_lookup instead |
| out_of_scope | 4/4 (100%) | |

**Known boundary collapses:**
1. `schedule_query` ↔ `single_course_lookup`: exam-timing queries route to single_course_lookup. Matters because schedule_query forces a SCHEDULE category filter; single_course_lookup does not.
2. `general_academic` ↔ `single_course_lookup`: any query mentioning a specific course ID routes to single_course_lookup. Functionally better (filtered), but means general_academic only fires for course-discovery queries with no course ID. Is this actually correct taxonomy behavior?

**Multi-course comparison whitespace bug:** Near-duplicate chunks from multiple sections of the same course (e.g., 3 sections of CSCE 120 all have near-identical GRADING content) are passed to the generator unfiltered. Gemini pads markdown table cells with thousands of spaces, wasting token budget and truncating the response.

### Current Router Prompt (exact)

```
You are a query classifier for a Texas A&M University course assistant.
Given the user's question, extract structured JSON with these fields:

{
  "intent": one of [...8 intents...],
  "course_ids": list of course IDs mentioned (e.g. ["CSCE 120", "CSCE 221"]), or [],
  "section": section number if mentioned, or null,
  "category": one of [11 categories] if relevant, or null,
  "policy_name": boilerplate policy name if asking about a specific policy, or null,
  "rewritten_query": the user's question rewritten for optimal search retrieval,
  "confidence": float 0-1
}

Intent definitions:
- "single_course_lookup": asking about a specific course's details (grading, schedule, instructor, materials, AI policy, etc.)
- "multi_course_comparison": comparing two or more courses or sections
- "aggregation_query": asking for counts, lists, or summaries across sections
- "policy_lookup": asking about a university-wide boilerplate policy
- "schedule_query": asking when a course meets, meeting times, days, room locations
- "instructor_query": asking who teaches a course, office hours, contact info
- "general_academic": broad questions not tied to a specific section
- "out_of_scope": greetings, weather, unrelated questions

Query rewriting rules:
- "late work" → "attendance makeup deadline extensions late submission"
- "ChatGPT" / "AI tools" → "AI policy artificial intelligence generative AI tools"
- "prereqs" → "prerequisites required courses corequisites"
- "prof" / "teacher" → "instructor professor"
- "grade breakdown" → "grading policy grade distribution weight percentage"
```

Router settings: temperature=0, max_output_tokens=512, thinking_budget=512.

### Current Generator Prompts (exact)

**Base system (all intents):**
```
You are TamuBot, an academic assistant for Texas A&M University.
RULES:
1. Answer ONLY based on the provided <context>. Never invent information.
2. Cite sources using [Source N] notation.
3. If context is insufficient, say so clearly.
4. Do NOT answer questions outside TAMU academics.
5. Be concise but thorough. Use markdown formatting.
6. When using markdown tables, do NOT pad cells with extra spaces.
```

**Intent-specific addons (appended per intent):**
- single_course_lookup: "Provide a clear, detailed answer. Include the course ID and section."
- multi_course_comparison: "Present the comparison in a clear markdown table. Highlight key differences."
- aggregation_query: "Present numerical data clearly. List all relevant items."
- policy_lookup: "Provide the policy information accurately and completely."
- schedule_query: "Be precise about days, times, and locations."
- instructor_query: "Include name, office hours, email, and office location if available."
- general_academic: "Provide a helpful answer. If multiple courses are relevant, mention each."

Generator settings: temperature=0.2, max_output_tokens=4096.

### Context Format Passed to Generator

```xml
<context>
<chunk source="1" course="CSCE 120" section="500" category="GRADING" instructor="Calvin Beideman" term="Spring 2026">
<title>Grading Policy</title>
<content>... grading breakdown ...</content>
</chunk>
</context>
Question: What is the grading breakdown for CSCE 120?
```

### What I Need

**Part A — Per-Intent k Optimization**

For each of the 8 intents, recommend `retrieve_k` (MongoDB candidates) and `rerank_k` (chunks to generator), with justification:
- How many chunks does a correct answer actually require per intent?
- Optimal over-retrieval ratio for Voyage rerank-2? (RAG literature: 3-10x)
- What happens with too many chunks (context dilution, token cost, hallucination) vs. too few (incomplete answers)?
- Should we hard-filter to the target category before retrieval, or let the reranker sort it out from a broader set?

Data constraints: a single course section has 8-14 chunks max. Policy lookup bypasses chunks entirely. Aggregation queries hit the courses collection, not chunks. For multi-course comparison, retrieval runs independently per course — what's the right per-course k?

**Part B — Multi-Course Comparison Fix (highest priority)**

Evaluate these 5 options for fixing the near-duplicate chunk problem:

1. **Deduplicate by (course_id, category)** — keep only the top-scored chunk per pair before passing to generator. Simple, low latency.
2. **Semantic deduplication** — drop chunks with cosine similarity > 0.95 to an already-selected chunk. More principled, adds latency.
3. **Prompt engineering** — instruct generator to use bullet lists instead of markdown tables for comparisons.
4. **Section-level pre-aggregation** — merge chunks across sections of the same course at ingestion time (schema change).
5. **Switch generator to gemini-2.0-flash-lite** — may have different table rendering behavior and lower cost.

For each: pros, cons, implementation complexity, root cause vs. symptom fix. Recommend primary fix + secondary mitigation.

**Part C — Router Prompt Optimization**

Fix the two boundary collapses with concrete prompt changes:

1. **schedule_query vs. single_course_lookup boundary:** Exam schedule queries ("When is the CSCE 120 final exam?", "What time is the midterm?") misroute to single_course_lookup. What exact prompt change fixes this? Should we add few-shot examples? Expand the schedule_query definition?

2. **general_academic vs. single_course_lookup boundary:** Is the current behavior (route to single_course_lookup when a course ID is present) actually correct? When should general_academic fire for a named-course query? Propose a revised intent definition that resolves this ambiguity.

Also research:
- Chain-of-thought in router vs. direct JSON output: latency/accuracy tradeoff at thinking_budget=512
- Appropriate confidence threshold for fallback to broad hybrid search (currently 0.5)
- How to handle multi-turn context in the router without full session management (e.g., passing last user+assistant turn)

**Part D — Generator Prompt Optimization**

For each intent, assess the current addon prompt and recommend improvements:

1. **single_course_lookup:** When multiple sections of the same course appear in context, which section should the answer reference? How should the prompt handle this?
2. **multi_course_comparison:** Given the whitespace bug, should we switch from table to bullet-list format in the prompt? What exact instruction prevents the padding behavior?
3. **aggregation_query:** The context contains MongoDB aggregation output (JSON), not chunks. Should the prompt explain this different context format?
4. **policy_lookup:** Policies are 500-2000 words. Should the generator summarize, quote verbatim, or extract the most relevant section?
5. **schedule_query:** Should the prompt always require day + time + location + section in the response, even if the user asked for only one?
6. **instructor_query:** Many instructors redirect office hours to Canvas ("see Canvas link"). How should the generator handle indirect answers?
7. **general_academic:** Highest hallucination risk. Should temperature be 0 instead of 0.2? Should the prompt require explicit uncertainty statements when context is thin?
8. **Context compression:** With up to 5 chunks × 300-500 tokens each, should we pre-summarize long chunks before passing to the generator?

**Part E — Future: Conversational Session Management (Architecture Preview Only)**

Do not recommend implementing this now. Provide only a 1-page architectural sketch for a future phase.

Current gap: the system is stateless — each query is independent. Example: a student asks "How hard is CSCE 638?" The optimal answer depends on their math and CS background. Ideally the router should recognize this and ask a clarifying question ("What is your level in linear algebra? Have you taken CSCE 411?") before retrieving.

Research:
1. Minimum viable session context for a RAG chatbot (last N turns? structured user profile?)
2. How conversation history should flow to the router (full history vs. summary vs. last turn)
3. Signal for "ask clarifying question" vs. "retrieve now" — what triggers clarification?
4. How session context interacts with the current intent taxonomy
5. Token cost implications of adding N turns to the router prompt
6. Best open-source libraries for RAG session management (LangGraph, MemGPT, others)

### Key Constraints

- Free tier only (Google AI API free tier for Gemini Flash, Voyage AI free tier, MongoDB Atlas M0)
- No LangChain/LlamaIndex — raw SDK calls only
- Latency target: p90 total pipeline < 5 seconds
- Must scale from 259 sections now to 7,889 sections after full ingestion

### Output Format

1. **Per-intent k-value table** — retrieve_k and rerank_k for each of the 8 intents with rationale
2. **Multi-course comparison fix** — primary fix + fallback, with concrete code changes
3. **Revised router prompt** — complete rewritten ROUTER_PROMPT string, explain each change
4. **Revised generator addon prompts** — updated for each of the 8 intents
5. **Taxonomy reassessment** — keep 8 intents, merge some, or split others? Justify with the boundary collapse data.
6. **Implementation priority order** — max impact, minimum effort first
7. **Session management sketch** — 1-page future architecture design

---

## PROMPT 8: Systematic RAG Pipeline Optimization — Parameter Search and Experiment Infrastructure

I'm building **TamuBot**, a production RAG chatbot for Texas A&M University students built on Python 3.12, LangGraph, MongoDB Atlas, Voyage AI, and Gemini Flash. The pipeline is modular and has ~15 tunable parameters. I want to run a structured optimization study (~15–20 experiment runs) to find the parameter combination that maximizes answer quality per unit cost.

**My highest priority is cost/efficiency ratio** — I care about RAGAS quality (faithfulness + answer relevancy), latency, and API token spend simultaneously, not just accuracy in isolation.

### Parameters I'm tuning

**Retrieval:**
- `retrieve_k` per query function (hybrid_course: 10–40, semantic_general: 15–50, recurrent: 5–25)
- `rerank_k` per query function (typically retrieve_k / 3–4)
- `max_retrieve_k` global cap (40–80)

**Ingestion (requires re-embedding, higher cost per trial):**
- `chunk_size` (200–800 tokens)
- `chunk_overlap` (0–150 tokens)

**Context:**
- `chunks_per_slot` (1–4 chunks per (course, category) slot)
- `stratified_fallback_per_course` (3–10)

**Generation:**
- `thinking_budget` (0, 512, 1024, 2048)
- `temperature` (0.0, 0.1, 0.2, 0.3)
- `generation_model` (categorical: "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash")

**Reranking:**
- `rerank_model` (categorical: "rerank-2", "rerank-2-lite")

### Evaluation setup

Each run benchmarks against a 50-question golden set with 7 strata (factual lookup, advisory, multi-course comparison, semantic general, recurrent/multi-turn). Metrics captured per run:
- **RAGAS faithfulness** (0–1, higher = answer grounded in context)
- **RAGAS answer_relevancy** (0–1, higher = answer targets the question; currently 0.50, main gap)
- **Mean pipeline latency** (ms)
- **Estimated API token spend** (input + output tokens, for cost proxy)
- **Router accuracy** (ground truth function classification)
- **Citation pass rate**

A full run (50 questions, with RAGAS) costs ~$0.30–0.50 in API calls and takes ~20–30 minutes. Ingestion-requiring runs (chunk_size/overlap changes) add ~15 minutes and ~$0.10. Budget: ~20 runs total.

### What I want from this research

I am **not anchored to any particular optimization paradigm**. I want you to survey the landscape of approaches and recommend what actually works for this problem, including but not limited to:

- **Bayesian optimization / surrogate model methods** (Gaussian processes, TPE, random forests as surrogates) — Optuna, Ax, BoTorch, SMAC3
- **Design of Experiments** (Sobol sequences, Plackett-Burman, Latin hypercube screening)
- **Evolutionary / genetic algorithms** — CMA-ES, NSGA-II for multi-objective
- **Reinforcement learning / bandit approaches** — multi-armed bandit, contextual bandits, RL-based HPO
- **Gradient-free methods** — Nelder-Mead, Powell, COBYLA for low-budget black-box
- **RAG-specific optimization frameworks** — AutoRAG, DSPy optimizers, LlamaIndex eval loops, anything built for LLM pipeline tuning
- **Any hybrid or novel approach** I may not have considered

### Key constraints

1. **Sequential execution** — each run calls paid APIs, cannot parallelize (no distributed search)
2. **Budget: ~20 runs** — algorithm must be sample-efficient
3. **Mixed parameter types** — integers (k values), floats (temperature), categoricals (model names), and two-level parameters (chunk_size/overlap) that are expensive to change (require re-ingestion)
4. **Multi-objective** — I want to jointly optimize quality (RAGAS) AND efficiency (latency + cost), not just one metric
5. **Stateful / sequential** — ideally the system uses results from prior runs to intelligently choose the next configuration to try (not static grid or random search)
6. **Open-source preferred** — I want to understand and control the algorithm, not a black box SaaS
7. **Must handle parameter dependencies** — chunk_size/overlap only need to change infrequently (they require re-ingestion); the other params can be swept cheaply

### Output I want

1. **Ranked comparison table** of approaches on: sample efficiency (expected result quality at run budget), handling of mixed types, multi-objective support, sequential/adaptive capability, Python library maturity, ease of integration
2. **Recommended approach** with justification — what algorithm/library would you use for this exact setup, and why?
3. **Concrete workflow** — step by step: how to define the parameter space, how to generate the initial screening configs, how to observe results and update the model, how to recommend the next config
4. **Handling expensive parameters** — best practice for treating chunk_size/overlap as "outer loop" vs inner-loop parameters given re-ingestion cost
5. **Stopping criteria** — with only ~20 runs, how do I know when I've found a good enough optimum vs. need more runs?
6. **GitHub examples** — 2–3 real open-source repos that do something similar (LLM pipeline tuning, RAG optimization, or black-box HPO with limited budget)
7. **What to avoid** — common pitfalls in low-budget black-box optimization of noisy objectives (RAGAS scores have variance)
8. **References** — papers/repos on per-intent retrieval tuning, RAG deduplication, structured output prompt engineering
