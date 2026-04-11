---
name: probe-rag
description: Use when testing a query through the RAG pipeline, running a probe, checking retrieval or generation output, or inspecting a Langfuse trace after a code change
triggers: ["probe", "run probe", "test query", "check rag output", "inspect trace", "langfuse trace"]
---

# /probe — RAG Pipeline Probe

## Args

| Input | Flag |
|---|---|
| Query string | `--query "<string>"` |
| Test IDs (1-based) | `--test-ids 1 3 7` |
| Full suite | `--suite all` |
| Tag run | `--tag <label>` |
| RAGAS metrics | `--ragas` |
| Memory session | `--memory --thread-id <id>` |

If no args given, ask what to probe.

## Run

```bash
python evals/run_probe.py <args>
```

## Output per query

- Router function + derived `course_ids` / `categories`
- Chunks retrieved after reranking
- Gate 1 citation status (PASS / MISSING / N/A)
- Answer length (chars)
- Langfuse trace URL

## Langfuse MCP

If `langfuse` MCP server is active, offer to pull full trace details or compare two tagged runs. Gate 2 groundedness scores appear ~20–30s after run.

## Reference

- Script: `evals/run_probe.py`
- Test suite: `evals/eval_pipeline.py::TEST_SUITE`
