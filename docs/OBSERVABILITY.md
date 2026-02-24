# TamuBot Observability Runbook

How to monitor, interpret, and act on telemetry from the TamuBot RAG pipeline using Langfuse and RAGAS.

**Langfuse project:** https://cloud.langfuse.com/project/cmlyjfvy200qbad07ezy65y21

---

## 1. Where to Find Things in Langfuse

### Traces tab → individual request traces
**URL:** `/project/.../traces`

Each user query produces one trace named `TamuBot_Complete_Pipeline`. Click any row to open the detail view.

**What to look at in a trace:**
- **Input / Output** at the top — original question and final answer
- **Scores section** (below input/output) — RAGAS `faithfulness` and `answer_relevancy` (0–1 scale, appear ~10s after the answer was generated)
- **Timeline** — visual waterfall of all spans; hover each bar to see duration
- **Span tree** (left panel) — expandable hierarchy of all stages

### Scores tab → aggregate quality trends
**URL:** `/project/.../scores`

Plots all RAGAS scores across traces over time. Use this to spot regressions after code changes or data refreshes.

- Filter by **Score Name**: `faithfulness` or `answer_relevancy`
- Filter by **Date range** to compare before/after a deployment

### Generations tab → LLM cost and token tracking
**URL:** `/project/.../generations`

Every `Generator_Stage` generation appears here with model name, token counts, and latency.

- **Input tokens** = context XML + question sent to Gemini 2.0 Flash
- **Output tokens** = generated answer
- `thinking_tokens` is in the **Metadata** column (Gemini 2.5 Flash router only; generator has 0)

---

## 2. The Trace Hierarchy — What Each Span Means

```
TamuBot_Complete_Pipeline  (Trace)
├── Router_Stage           (Span)
│   └── metadata: intent, confidence, course_ids, rewritten_query,
│                 input_tokens, output_tokens, thinking_tokens
├── Retrieval_Stage        (Span)
│   ├── Voyage_Embeddings  (Span) — query → 1024-dim vector via Voyage voyage-3
│   ├── MongoDB_Hybrid_Search (Span) — Atlas $vectorSearch + $search → RRF fusion
│   │   └── metadata: n_vector_results, n_text_results, n_fused
│   └── Voyage_Reranker    (Span) — cross-encoder rerank-2 → top-k
│       └── metadata: n_returned, relevance_scores[], min_score, max_score
└── Generator_Stage        (Generation)
    ├── model: gemini-2.0-flash
    ├── usage: {input, output}
    └── metadata: intent, course_ids, n_sources, thinking_tokens
```

**Absent spans mean:**
- No `Retrieval_Stage` → intent was `out_of_scope` (no retrieval needed)
- No `Voyage_Reranker` → policy lookup returned a direct match (no reranking)
- `Router_Stage` exists but `Retrieval_Stage` doesn't → retrieval crashed (check `level: ERROR` on the span)

---

## 3. RAGAS Scores — How to Interpret Them

RAGAS scores appear on the trace ~5–15 seconds after the answer is rendered (background thread).

### Faithfulness (0 → 1)
Measures whether every factual claim in the answer is grounded in the retrieved context.

| Score | Meaning | Action |
|-------|---------|--------|
| 0.8 – 1.0 | Answer fully grounded — healthy | None |
| 0.5 – 0.8 | Some claims lack context support | Review the answer; check if retrieval returned relevant chunks |
| 0.0 – 0.5 | Significant hallucination risk | Investigate — did the generator invent information? Was context empty? |
| 0.0 exactly | No verifiable claims found | Often happens with very short or malformed answers (e.g., truncated tables) |

### Answer Relevancy (0 → 1)
Measures whether the answer actually addresses the user's question (embedding-based).

| Score | Meaning | Action |
|-------|---------|--------|
| 0.8 – 1.0 | Answer directly addresses the question | Healthy |
| 0.5 – 0.8 | Answer partially addresses the question | Check if intent was correctly classified |
| 0.0 – 0.5 | Answer does not match the question | Likely wrong intent or retrieval failure |

### When scores are missing
- Score missing entirely → RAGAS evaluation failed (check server logs for `RAGAS evaluation failed:`)
- `answer_relevancy = NaN` → embedding call failed; check Voyage AI API key and quota

---

## 4. Routine Monitoring Checklist

### After each significant code change
1. Open **Traces** → run 3–5 representative queries (one per intent type)
2. Verify the span tree is complete for each trace (all 5 spans present)
3. Check **Scores** after ~30s — confirm both metrics are populated, not NaN
4. Compare average scores to baseline in the **Scores** tab

### After a data refresh (re-ingestion)
1. Run one query per intent type
2. Check `Retrieval_Stage → MongoDB_Hybrid_Search → n_fused` — should be > 0 for all retrieval intents
3. Check `Retrieval_Stage → Voyage_Reranker → max_score` — should be > 0.7 for on-topic queries
4. Watch for faithfulness drop — near-duplicate chunks from new data can confuse the generator

### Weekly
1. **Scores tab** → filter last 7 days → check for score trend (rising or falling)
2. **Generations tab** → sort by input tokens descending — flag any traces consuming > 3,000 input tokens (context bloat)
3. Look for traces with `level: ERROR` in any span — indicates API failures

---

## 5. Diagnosing Common Problems

### "Traces not appearing in Langfuse"
- Check that `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in `.env`
- Look for `Langfuse REST client initialised.` in the server logs on startup
- Look for `Langfuse flush failed` warnings in logs — usually a network timeout
- Note: traces appear after `lf.flush()` is called (at end of each response), not during generation

### "RAGAS scores not appearing"
- RAGAS runs in background — wait 10–30s then refresh the trace page
- If still missing, check server logs for `RAGAS evaluation failed:`
- Common causes: Voyage AI rate limit, Google API quota, or very short answer (< 10 words)

### "faithfulness = 0.0 on every query"
- Usually means the generator is producing answers without verifiable factual claims
  - Truncated markdown tables (comparison query whitespace bug)
  - Empty context passed to generator (retrieval failed)
  - Canned `out_of_scope` response (no claims to verify — expected)
- Check `Generator_Stage → n_sources` in metadata — if 0, retrieval returned nothing

### "answer_relevancy = NaN"
- The Voyage AI embedding call for RAGAS failed
- Check Voyage AI API key is valid and has quota remaining
- Check for `GoogleGenerativeAIError` in logs — means wrong embedding model was called

### "Router_Stage shows wrong intent"
- Open the trace → Router_Stage → output → inspect the raw JSON from Gemini
- Check `confidence` — if < 0.5, system falls back to broad hybrid search (expected behavior)
- Add the misclassified query to `scripts/eval_pipeline.py` test suite for regression tracking

### "Retrieval_Stage missing or empty results"
- Open trace → Retrieval_Stage → MongoDB_Hybrid_Search → metadata
- `n_fused = 0` means no chunks matched — either MongoDB is empty or filters are too narrow
- Verify ingestion: `python -m db.ingest --dry-run` to preview what would be loaded

---

## 6. Key Metrics at a Glance

| Metric | Healthy | Warning | Action needed |
|--------|---------|---------|---------------|
| Faithfulness | > 0.7 | 0.4 – 0.7 | < 0.4 |
| Answer Relevancy | > 0.7 | 0.4 – 0.7 | < 0.4 |
| Router confidence | > 0.8 | 0.5 – 0.8 | < 0.5 (fallback mode) |
| Reranker max_score | > 0.7 | 0.4 – 0.7 | < 0.4 (poor retrieval) |
| n_fused results | > 5 | 2 – 5 | 0 (empty DB or bad filter) |
| Generator input tokens | < 2,000 | 2,000 – 3,500 | > 3,500 (context bloat) |
| Total trace latency | < 5s | 5 – 10s | > 10s |

---

## 7. Implementation Reference

| File | Role |
|------|------|
| `db/observability.py` | `MinimalLangfuseClient` (REST-based, Python 3.14 safe) + `compute_ragas_metrics()` + `run_ragas_background()` |
| `db/router.py` | Creates `Router_Stage` and `Retrieval_Stage` spans; threads them to search/reranker |
| `db/search.py` | Creates `Voyage_Embeddings` and `MongoDB_Hybrid_Search` child spans |
| `db/reranker.py` | Creates `Voyage_Reranker` child span with relevance scores |
| `db/generator.py` | Creates `Generator_Stage` generation with token usage mapping |
| `app.py` | Creates parent trace, calls `lf.flush()`, triggers `run_ragas_background()` |
| `config.py` | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` |

### Why a custom REST client instead of the Langfuse SDK?
The official `langfuse` Python SDK (all versions 2.x and 3.x) uses a Fern-generated API layer that depends on `pydantic.v1`. This breaks at import time on Python 3.14+ due to incompatible annotation evaluation changes (PEP 649). The custom client in `db/observability.py` posts directly to `/api/public/ingestion` and `/api/public/scores` using `httpx` — no pydantic.v1 dependency.

If a future Langfuse SDK version fixes Python 3.14 support, you can replace `MinimalLangfuseClient` with the official SDK — the interface (`trace()`, `span()`, `generation()`, `end()`, `flush()`, `score()`) is intentionally kept compatible.
