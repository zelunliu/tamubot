# CLAUDE.md

> Module-level detail: `rag/CLAUDE.md`, `ingestion_pipeline/CLAUDE.md`, `evals/CLAUDE.md`

## Docker Sandbox (standard dev entry point)

```bash
make sandbox-up      # start claude + api-proxy + app containers
make sandbox-shell   # open bash inside claude container
make sandbox-down    # tear down all containers
```

Streamlit: http://localhost:8501
Docs: `docs/DOCKER_SETUP.md` (Windows 11 WSL2 + Mac) | `docs/API_SAFETY.md` (proxy + rate limits)

## Commands

```bash
# Inside container (no Docker available) — bare host / WSL2 direct
streamlit run app.py --server.headless true          # start app

# Windows Git Bash (outside container, with venv)
source .venv/Scripts/activate && streamlit run app.py

make test | lint | typecheck | format | eval-router | probe | probe-full
```


## Gotchas

- **Config**: always `import config` in `rag/` — never `os.getenv()` directly
- **TAMU AI gateway** (`TAMU_API_KEY` set → `USE_TAMU_API=True`): always returns SSE regardless of `stream` param → ALL calls must use `stream=True` + `"".join(chunk.choices[0].delta.content or "" for chunk in stream)`. Base URL: `https://chat-api.tamu.ai/openai` (no `/v1`). Min `max_tokens=4096` or response is empty.

## Skills — Auto-Engage

Invoke via the Skill tool automatically (no `/` command needed) when intent matches:

- **probe-rag**: user asks to test a query, run a probe, check RAG output, or inspect a Langfuse trace
- **scrape**: user asks to scrape a site, download syllabi, or add/run a crawler
- **chunk-syllabi**: user asks to re-chunk syllabi, run chunking experiments, or inspect token-level chunk output
- **run-eval**: user asks to run eval, run benchmark, benchmark the pipeline, benchmark RAG, or run chunking eval
- **golden-set**: user asks to create, generate, refine, or export a golden set / eval set
- **server-ops**: user says "restart localhost/server/app", "start/stop server", "kill the server", "server status", "clear cache", or any variant of managing the local dev server
- **task-budget**: any task involving TAMU API, Voyage AI, or Google AI calls (RAG queries, probes, ingestion, benchmarks)
- **research-prompts**: user asks to generate or write a research prompt

When skill tool engaged, make sure to notify user!

