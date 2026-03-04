# /probe — RAG Pipeline Probe

Run one or more queries through the full TamuBot RAG pipeline (router → retrieval → rerank → generate → Gate 1 → Gate 2) with Langfuse tracing, then inspect the results.

## When invoked

When the user types `/probe` (with or without arguments), do the following:

### Step 1 — Determine what to run

Parse any arguments the user provided:
- A plain string → treat as `--query "<string>"`
- Numbers → treat as `--test-ids <numbers>`
- `all` → treat as `--suite all`
- `--tag <value>` → pass through as `--tag`
- `--ragas` → pass through as `--ragas`

If no arguments were provided, ask:
> "What would you like to probe? Provide a query string, test IDs (e.g. `1 3 7`), or `all` for the full suite. Optionally add `--tag <label>` or `--ragas`."

### Step 2 — Run the probe script

Execute the constructed command:
```bash
python evals/run_probe.py <args>
```

Wait for it to complete and capture the full stdout output.

### Step 3 — Summarize the results

For each query that ran, print:
- The router function and derived course_ids / categories
- Number of chunks retrieved after reranking
- Gate 1 citation status (PASS / MISSING / N/A)
- Answer length in characters
- The Langfuse trace URL (clickable)

### Step 4 — If Langfuse MCP is configured, offer deeper inspection

If the `langfuse` MCP server is active, offer to:
- Pull the full trace details (spans, generation text, scores)
- Compare two tagged runs side-by-side (e.g. `before` vs `after` a prompt change)
- Show Gate 2 groundedness scores once they appear (~20–30s after the run)

Say: "Langfuse MCP is available — I can pull trace details directly. Want me to inspect the trace for any of these queries?"

## Examples

```
/probe What is the grading breakdown for CSCE 638?
/probe 1 3 7
/probe all --tag generator_v2
/probe What topics are covered in CSCE 638? --ragas
```

## Reference

- Script: `evals/run_probe.py`
- Test suite: `evals/eval_pipeline.py::TEST_SUITE` (use `--test-ids` for 1-based IDs)
- Langfuse dashboard: `https://cloud.langfuse.com`
- Trace URL format: `{LANGFUSE_BASE_URL}/trace/{trace_id}`
