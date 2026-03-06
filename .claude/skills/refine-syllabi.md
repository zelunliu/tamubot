# /refine-syllabi — Syllabus quality audit and prompt refinement

**Goal**: efficient, concise parsing — no boilerplate leakage, no chunk bloat, no info loss, high-quality COURSE_SUMMARY.

**Flow**: Load history → Audit → Diagnose → Propose → **wait for approval** → Apply → Log → Reprocess → Report.
Never auto-cycle. User controls each iteration. Progress persists across sessions.

---

## Step 0 — Load history

Before doing anything, read the persistent log:

```bash
cd /c/dev/TAMU_NEW && python - <<'PYEOF'
import json
from pathlib import Path

log = Path("tamu_data/logs/refine_history.jsonl")
if not log.exists():
    print("No history yet — first run.")
else:
    entries = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"History: {len(entries)} cycle(s)\n")
    for e in entries:
        print(f"  Cycle {e['cycle']} | {e['timestamp'][:10]} | {len(e['applied'])} fix(es) applied | files: {', '.join(e['files_reprocessed'])}")
        for fix in e["applied"]:
            print(f"    [APPLIED] {fix['issue_type']}: {fix['description'][:80]}")
        for iss in e.get("open_issues", []):
            print(f"    [OPEN]    {iss['issue_type']}: {iss['description'][:80]}")
PYEOF
```

Use this to:
- Know what cycle we're on (next = max cycle + 1)
- Know which issue types have been fixed before
- Detect regressions: if an issue type appears that was marked APPLIED in a prior cycle → **flag it as REGRESSION**, propose a stronger fix (post-processing instead of prompt)

---

## Step 1 — Select scope

| Input | Scope |
|---|---|
| `CSCE 614 670` | audit those specific JSONs |
| `--department CSCE` | audit all CSCE JSONs in OUTPUT_DIR |
| `--all` / no args | audit all JSONs in OUTPUT_DIR |

OUTPUT_DIR = `tamu_data/processed/gem_parsed_<today>/`

---

## Step 2 — Run the audit script

```bash
cd /c/dev/TAMU_NEW && source .venv/Scripts/activate && PYTHONIOENCODING=utf-8 python - <<'PYEOF'
import json, sys
from pathlib import Path
import fitz
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

out_dir = Path("tamu_data/processed/gem_parsed_20260306")  # FILLED IN BY CLAUDE
raw_root = Path("tamu_data/raw")
fallbacks = sorted(raw_root.glob("simple_syllabus_*/"), reverse=True)

def find_pdf(stem):
    p = raw_root / "syllabi" / f"{stem}.pdf"
    if p.exists(): return p
    for d in fallbacks:
        c = d / f"{stem}.pdf"
        if c.exists(): return c
    return None

BOILERPLATE_PATTERNS = {
    "IT_HELPDESK":        "technology services (it) - main campus",
    "CANVAS_SUPPORT":     "canvas lms technical support",
    "STUDENT_RULE_7":     "work submitted by a student as makeup work for an excused absence is not considered late work",
    "DOC_HEADER":         "college of engineering\ncomputer science and engineering",
    "MEETING_TIMES_DUMP": "meeting times:\u00a0meeting type:",
    "ADA_BOILERPLATE":    "texas a&m university is committed to providing equitable access",
    "HONOR_CODE_VERBOSE": "an aggie does not lie, cheat or steal",
}

TEMPLATE_NOISE = {
    "MATERIALS_LABEL":     ("MATERIALS",        "this material is:"),
    "OUTCOMES_PREAMBLE":   ("LEARNING_OUTCOMES","upon completion of this course, the learner will be able to:"),
    "PREREQ_LABEL":        ("PREREQUISITES",    "prerequisite/corequisite(s):"),
    "MEETING_IN_OVERVIEW": ("COURSE_OVERVIEW",  "meeting times:"),
}

target_stems = [
    # FILLED IN BY CLAUDE — or empty for all
]
if not target_stems:
    target_stems = [jf.stem for jf in sorted(out_dir.glob("*.json"))]

file_stats = {}
boilerplate_hits = {}   # key -> [(stem, cat, excerpt)]
noise_hits = {}         # key -> [stem]

for stem in target_stems:
    jf = out_dir / f"{stem}.json"
    if not jf.exists(): continue
    with open(jf, encoding="utf-8") as f:
        d = json.load(f)
    if "error" in d: continue

    chunks = d.get("chunks", [])
    chunk_by_cat = {c["category"]: c for c in chunks}
    flags = []

    # Boilerplate leaks
    for chunk in chunks:
        cl = chunk["content"].lower()
        for key, pattern in BOILERPLATE_PATTERNS.items():
            if pattern in cl:
                boilerplate_hits.setdefault(key, []).append((stem, chunk["category"], chunk["content"][:120]))

    # Template noise
    for key, (cat, pattern) in TEMPLATE_NOISE.items():
        if cat in chunk_by_cat and pattern in chunk_by_cat[cat]["content"].lower():
            noise_hits.setdefault(key, []).append(stem)

    # INSTRUCTOR
    inst = chunk_by_cat.get("INSTRUCTOR", {})
    if not inst: flags.append("MISSING_INSTRUCTOR")
    elif len(inst.get("content","")) < 30: flags.append(f"THIN_INSTRUCTOR({len(inst['content'])}c)")

    # COURSE_OVERVIEW header leak
    ov = chunk_by_cat.get("COURSE_OVERVIEW", {})
    if ov:
        fl = ov.get("content","").strip().splitlines()[0].lower()
        if any(x in fl for x in ["college of engineering", "meeting times", "meeting type"]):
            flags.append("OVERVIEW_HEADER_LEAK")

    # COURSE_SUMMARY quality
    cs = chunk_by_cat.get("COURSE_SUMMARY", {})
    if cs:
        content = cs.get("content","")
        topic_line = next((l for l in content.splitlines() if l.startswith("Topics:")), "")
        topic_count = len(topic_line.replace("Topics:","").split(",")) if topic_line else 0
        if len(content) < 300: flags.append(f"SUMMARY_SHORT({len(content)}c)")
        if topic_count < 5: flags.append(f"SUMMARY_FEW_TOPICS({topic_count})")

    # Coverage
    pdf_path = find_pdf(stem)
    cov = None
    if pdf_path:
        doc = fitz.open(str(pdf_path))
        pdf_text = "\n".join(p.get_text() for p in doc)
        doc.close()
        all_chunk_text = "\n".join(c["content"] for c in chunks)
        cov = round(len(all_chunk_text) / max(len(pdf_text), 1) * 100, 1)
        if cov < 15: flags.append(f"LOW_COVERAGE({cov}%)")

    char_total = sum(len(c["content"]) for c in chunks)
    file_stats[stem] = {"flags": flags, "chars": char_total, "coverage": cov, "cats": list(chunk_by_cat)}

# Output
print("=== PER-FILE ===")
for stem, s in file_stats.items():
    cov = f"{s['coverage']}%" if s['coverage'] else "?"
    flags = ", ".join(s["flags"]) if s["flags"] else "OK"
    print(f"  {stem[-32:]:<34} {s['chars']:>6}c  cov={cov:<6}  {flags}")

print("\n=== BOILERPLATE LEAKS ===")
for key, hits in sorted(boilerplate_hits.items()):
    print(f"  [{key}] {len(hits)} file(s): {', '.join(set(s for s,_,_ in hits))}")
    print(f"    e.g. {hits[0][1]}: {repr(hits[0][2][:80])}")

print("\n=== TEMPLATE NOISE ===")
for key, stems in noise_hits.items():
    cat, pattern = TEMPLATE_NOISE[key]
    print(f"  [{key}] '{pattern}' in {len(stems)} file(s)")

print(f"\n=== TOTALS ===")
print(f"  Files: {len(file_stats)} | With flags: {sum(1 for s in file_stats.values() if s['flags'])}")
print(f"  Boilerplate leak types: {len(boilerplate_hits)} | Noise types: {len(noise_hits)}")
PYEOF
```

---

## Step 3 — Diagnose with history awareness

Cross-reference audit output against history (Step 0). Classify each issue:

| Classification | Meaning | Approach |
|---|---|---|
| **NEW** | First time seen | Prompt tweak |
| **REGRESSION** | Was fixed in a prior cycle but returned | Escalate to post-processing in `parse_pdf()` |
| **PERSISTENT** | Has appeared in multiple cycles, never fixed | Escalate to post-processing or mark as known/acceptable |
| **KNOWN-ACCEPTABLE** | e.g. thin SCHEDULE because external doc | Document, don't fix |

---

## Step 4 — Propose changes (numbered list)

Present every finding as a numbered proposal. Format:

```
[N] SEVERITY — ISSUE_TYPE (classification) — N files affected
    What: <one line describing the problem>
    Example: <stem> / <category>: "<excerpt>"
    Fix type: PROMPT / POST-PROCESSING / INVESTIGATE
    Proposed change: <exact before → after for PROMPT, or description for code>
    Reprocess: <N files or "all">
```

Severity: **CRITICAL** (data loss / wrong data) | **WARNING** (boilerplate leaking) | **INFO** (minor noise)

**Do not apply anything yet.** End with:
> "Which changes to apply? Reply with numbers (e.g. `1 3`), `all`, or `none`. I'll show you the exact edits before touching any file."

---

## Step 5 — Show exact diffs, then apply

For each approved item, show:
```
PROMPT change [N]:
  OLD: "..."
  NEW: "..."
```
Then ask: "Apply these N change(s)?"

On confirmation, use Edit tool — minimal, targeted changes only.

---

## Step 6 — Reprocess affected files

Ask: "Reprocess N affected files?" On yes: delete their JSONs, run the pipeline (same as `/process-syllabi` Step 2a). After completion show per-file before/after summary (chunk sizes, flags, coverage).

Then run `python ingestion_pipeline/rebuild_csv.py`.

---

## Step 7 — Write history entry

After every apply+reprocess, append to `tamu_data/logs/refine_history.jsonl`:

```bash
cd /c/dev/TAMU_NEW && python - <<'PYEOF'
import json
from datetime import datetime
from pathlib import Path

Path("tamu_data/logs").mkdir(parents=True, exist_ok=True)
log = Path("tamu_data/logs/refine_history.jsonl")

# Load previous entries to get cycle number
prev = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()] if log.exists() else []
cycle = (max(e["cycle"] for e in prev) + 1) if prev else 1

entry = {
    "cycle": cycle,
    "timestamp": datetime.now().isoformat(),
    "scope": "FILLED_IN_BY_CLAUDE",           # e.g. "CSCE 614 670" or "--all"
    "files_reprocessed": [],                   # FILLED IN BY CLAUDE — list of stems
    "applied": [                               # FILLED IN BY CLAUDE
        # {"issue_type": "IT_HELPDESK", "classification": "NEW", "description": "...", "fix_type": "PROMPT"}
    ],
    "open_issues": [                           # issues found but NOT applied this cycle
        # {"issue_type": "...", "description": "...", "reason_skipped": "user chose not to"}
    ],
    "prompt_snapshot_hash": "",                # FILLED IN BY CLAUDE — sha1 of PROMPT constant after changes
}

with open(log, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry) + "\n")
print(f"Logged cycle {cycle} -> {log}")
PYEOF
```

To get the PROMPT hash:
```bash
cd /c/dev/TAMU_NEW && python -c "
import hashlib, sys
sys.path.insert(0,'.')
from ingestion_pipeline.process_syllabi import PROMPT
print(hashlib.sha1(PROMPT.encode()).hexdigest()[:12])
"
```

---

## What good looks like

| Metric | Target |
|---|---|
| COURSE_SUMMARY | 400–1000c, ≥8 topics, no narrative phrases |
| GRADING | 200–5000c, grade weights present |
| COURSE_OVERVIEW | Starts from catalog description, no header/meeting dump |
| INSTRUCTOR | ≥30c, name + email minimum |
| Coverage | 30–60% (boilerplate legitimately excluded) |
| Boilerplate leaks | 0 |
| Template noise labels in content | 0 |

## Known boilerplate — never in chunks

| Pattern | Resolution |
|---|---|
| "Technology Services (IT) - Main Campus" + Canvas block | `boilerplate_policies` |
| "Work submitted by a student as makeup work..." (Student Rule 7) | `boilerplate_policies` |
| "College of Engineering / CSCE XXX Syllabus" header | strip (already in metadata) |
| "This material Is: Required/Recommended/Optional" | strip (TAMU template label) |
| "Upon completion of this course, the learner will be able to:" | strip preamble, keep outcomes list |
| "Prerequisite/Corequisite(s):" label | strip label, keep course codes |
| "Meeting Times: Meeting Type: LEC..." block | strip (already in course_metadata) |
| ADA, FERPA, Title IX, Honor Code, Nondiscrimination, Mental Health full text | `boilerplate_policies` |
