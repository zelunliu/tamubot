---
name: research-prompts
description: Use when generating or writing a deep-research prompt for the TamuBot RAG pipeline
triggers: ["research prompt", "generate prompt", "write research prompt", "create research prompt"]
---

# /research-prompt — Generate Research Prompt

Append a numbered deep-research prompt to `docs/research_prompts.md`.

## Steps

**1 — Read context**
Read latest 2–3 eval reports (`tamu_data/evals/reports/*.md`) and source files relevant to the topic.

**2 — Get prompt number**
```bash
grep -c "^## PROMPT" docs/research_prompts.md
```
New number = count + 1.

**3 — Write prompt** using this structure:

```markdown
---

## PROMPT N: <Title>

### Goal
<One sentence: what we're optimizing and why.>

### Current State
<Only relevant numbers/facts — latency, RAGAS scores, token counts, prompt sizes.>

### Research Questions
<Specific, answerable questions grouped by subtopic.>

### Deliverables

1. Compare 2–3 concrete approaches with trade-offs (latency, quality, complexity)
2. ~2-page implementation summary for an AI coding agent:
   - What to change (file, component, technique)
   - Expected impact
   - Risks / what not to break
```

**4 — Insert** at the top of `docs/research_prompts.md`, immediately after the 5-line preamble ending in `---`.

Confirm: "PROMPT N written to `docs/research_prompts.md`. Ready to copy into your research chat."

## Reference

- `docs/research_prompts.md` — target file
- `tamu_data/evals/reports/` — eval data
- `rag/prompts.py`, `rag/v4/nodes/`, `rag/v4/components/` — pipeline source
