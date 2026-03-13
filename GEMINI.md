# GEMINI.md — TamuBot Project
Be concise and direct with your answers. Before any action tell me what you plan to do, which files will be affected and ask clarifying questions if needed.

## Project
RAG chatbot for Texas A&M course/syllabus queries. Syllabi are preprocessed into chunked JSON files used for vector search.

## Operational Mandates
- **State Intent:** You must provide a concise, one-sentence explanation of your intent or strategy immediately before executing any tool calls.
- **Security First:** Never log, print, or commit secrets. Protect `.env` files.
- **Ignored Files:** You must use source PDFs and other files listed in `.gitignore` or otherwise restricted, using shell commands or other appropriate means. Do not substitute these files with processed versions (like Markdown) unless specifically requested.
- **No Assumptions:** Never take significant actions or move to a different file type (e.g., from PDF to Markdown) based on assumptions or tool limitations. If a file is restricted, find a way to access the original source as required by this mandate.

## Data Integrity & Safety Mandates

### 1. Mandatory Backup for Ignored Files
- **Rule:** Before any destructive write (overwriting or deleting) on a file listed in `.gitignore` or otherwise restricted, you MUST create a backup with a `.bak` extension.
- **Goal:** Provide a local "undo" mechanism for files not version-controlled.
- **Check:** If a `.bak` file already exists, append a timestamp (e.g., `filename.20260313.bak`) to prevent overwriting an existing backup.

### 2. Structural Verification for Binary/Structured Formats
- **Rule:** Before writing to any non-plain-text file (Excel, SQLite, PDF, JSON), you MUST first use a tool to inspect its internal structure.
  - **Excel:** List all sheet names using a library like `openpyxl`.
  - **SQLite:** List all table names and schema.
  - **JSON:** Check for top-level keys if the intent is only to update a subset.
- **Goal:** Prevent "blind" overwrites that accidentally strip hidden data (like extra sheets or formatting).
- **Mandate:** If inspection reveals more data than the intended edit (e.g., multiple sheets when only one is being updated), you MUST use a surgical library (like `openpyxl` for Excel) instead of a "full-save" tool (like `pandas.to_excel`).

### 3. Surgical Edits over Full Overwrites
- **Rule:** Prefer targeted updates (e.g., updating one row, cell, or JSON key) over reading the entire file into memory and writing it back out.
- **Goal:** Minimize risk of formatting loss, encoding issues, or accidental data stripping.

## Commands (Imported from CLAUDE.md)
```bash
source .venv/Scripts/activate && streamlit run app.py   # Windows Git Bash
make test | lint | typecheck | format | eval-router | probe | probe-full
```

## Gotchas (Imported from CLAUDE.md)
- **Config**: always `import config` in `rag/` — never `os.getenv()` directly
- **TAMU AI gateway** (`TAMU_API_KEY` set → `USE_TAMU_API=True`): always returns SSE regardless of `stream` param → ALL calls must use `stream=True` + `"".join(chunk.choices[0].delta.content or "" for chunk in stream)`. Base URL: `https://chat-api.tamu.ai/openai` (no `/v1`). Min `max_tokens=4096` or response is empty.
- **Gemini JSON mode**: with `response_mime_type="application/json"` + schema, free-form Markdown fields silently return empty — always render Markdown in Python from structured data
- **Langfuse SDK / Python 3.14**: `pydantic.v1` incompatible → custom `MinimalLangfuseClient` in `rag/observability.py`; revert when SDK ships fix
- **ingestion_pipeline**: stays on direct `GOOGLE_API_KEY` — PDF multimodal (`Part.from_bytes`) not supported by TAMU gateway

## Skills & Superpowers
This project uses the **Superpowers** system. Invoke specific skills via `activate_skill(name='...')` for specialized workflows.

**Available Skills:**
- **qa-protocol**: Standard Preprocessing QA Protocol for course syllabi. Located at `.gemini/skills/qa-protocol/SKILL.md`.
- **using-superpowers**: Establishes how to find and use skills.
- **brainstorming**: Use before starting any creative work or major refactoring.
- **writing-plans**: Use when you have a spec or requirements, before touching code.
- **verification-before-completion**: Use before claiming work is complete.

## Key Folders
| Folder | Contents |
|--------|----------|
| `tamu_data/processed/v2_step0_source/` | Source PDFs (versioned, e.g. `*_v006.pdf`) |
| `tamu_data/processed/v2_step1_markdown/` | Font-annotated Markdown extracted from PDFs |
| `tamu_data/processed/v2_step2_boilerplate/` | Markdown after institutional boilerplate is stripped |
| `tamu_data/processed/v3_step3_flat/` | Final chunked JSON files (one per syllabus) |
