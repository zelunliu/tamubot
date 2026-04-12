# Excel-Native Golden Sets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all JSONL golden set files with Excel, add a shared `evals/golden_set.py` module, and append per-run results as columns in the golden set file.

**Architecture:** New `evals/golden_set.py` owns all Excel I/O (`load`, `save`, `append_run_column`). `eval_chunking.py` and `run_benchmark.py` swap their JSONL loaders for `golden_set.load()` and call `append_run_column()` after each run. A migration script converts existing JSONL → Excel and deletes JSONL files.

**Tech Stack:** Python, openpyxl

---

## File Map

| Action | File | Purpose |
|---|---|---|
| Create | `evals/golden_set.py` | `load()`, `save()`, `append_run_column()` |
| Create | `tests/test_golden_set.py` | Unit tests for the module |
| Create | `evals/migrate_golden_sets.py` | One-off migration script |
| Modify | `evals/eval_chunking.py` | Swap loader, add run column |
| Modify | `evals/run_benchmark.py` | Swap loader, add run column, handle missing stratum/source_course_id |
| Modify | `evals/import_eval_draft.py` | Write `.xlsx` instead of JSONL |
| Modify | `skills/run-eval/SKILL.md` | `*.jsonl` → `*.xlsx` |

---

## Task 1: Create `evals/golden_set.py`

**Files:**
- Create: `evals/golden_set.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_golden_set.py`:

```python
"""Tests for evals/golden_set.py — load, save, append_run_column."""
import json
from pathlib import Path
import pytest


SAMPLE_ITEMS = [
    {"id": 1, "question": "What is CSCE 611 about?", "reference_answer": "OS course.",
     "expected_function": "hybrid_course", "human_notes": None},
    {"id": 2, "question": "What courses cover ML?", "reference_answer": "CSCE 638.",
     "expected_function": "semantic_general", "human_notes": "reworked"},
]


def test_save_then_load_roundtrip(tmp_path):
    from evals.golden_set import load, save
    path = tmp_path / "test.xlsx"
    save(SAMPLE_ITEMS, path)
    loaded = load(path)
    assert len(loaded) == 2
    assert loaded[0]["question"] == "What is CSCE 611 about?"
    assert loaded[0]["expected_function"] == "hybrid_course"
    assert loaded[0]["human_notes"] is None
    assert loaded[1]["human_notes"] == "reworked"


def test_load_skips_empty_question_rows(tmp_path):
    from evals.golden_set import load, save
    items = SAMPLE_ITEMS + [
        {"id": 3, "question": "", "reference_answer": "", "expected_function": "", "human_notes": None}
    ]
    path = tmp_path / "test.xlsx"
    save(items, path)
    loaded = load(path)
    assert len(loaded) == 2


def test_load_ignores_run_columns(tmp_path):
    import openpyxl
    from evals.golden_set import save, load
    path = tmp_path / "test.xlsx"
    save(SAMPLE_ITEMS, path)

    # manually append a run column
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    ws.cell(row=1, column=6, value="run:my_experiment")
    ws.cell(row=2, column=6, value="CSCE 611 §SCHEDULE 0.92")
    wb.save(path)

    loaded = load(path)
    assert "run:my_experiment" not in loaded[0]
    assert len(loaded[0]) == 5  # exactly schema keys


def test_append_run_column_adds_column(tmp_path):
    from evals.golden_set import save, append_run_column
    import openpyxl
    path = tmp_path / "test.xlsx"
    save(SAMPLE_ITEMS, path)

    append_run_column(path, "exp_20260412", {1: "CSCE 611 §SCHEDULE 0.92", 2: "CSCE 638 §LO 0.87"})

    wb = openpyxl.load_workbook(path)
    ws = wb.active
    headers = [c.value for c in ws[1]]
    assert "run:exp_20260412" in headers
    col_idx = headers.index("run:exp_20260412") + 1
    assert ws.cell(row=2, column=col_idx).value == "CSCE 611 §SCHEDULE 0.92"
    assert ws.cell(row=3, column=col_idx).value == "CSCE 638 §LO 0.87"


def test_append_run_column_overwrites_existing(tmp_path):
    from evals.golden_set import save, append_run_column
    import openpyxl
    path = tmp_path / "test.xlsx"
    save(SAMPLE_ITEMS, path)

    append_run_column(path, "exp_20260412", {1: "old value"})
    append_run_column(path, "exp_20260412", {1: "new value"})

    wb = openpyxl.load_workbook(path)
    ws = wb.active
    headers = [c.value for c in ws[1]]
    col_idx = headers.index("run:exp_20260412") + 1
    assert ws.cell(row=2, column=col_idx).value == "new value"
    # only one run column added
    assert headers.count("run:exp_20260412") == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_golden_set.py -v
```

Expected: `ModuleNotFoundError: No module named 'evals.golden_set'`

- [ ] **Step 3: Implement `evals/golden_set.py`**

```python
"""Shared golden set I/O — Excel-native, no JSONL.

Schema columns (in order):
    id, question, reference_answer, expected_function, human_notes

run:<experiment> columns are appended by eval scripts and ignored by load().
"""
from __future__ import annotations

from pathlib import Path

SCHEMA_COLUMNS = ["id", "question", "reference_answer", "expected_function", "human_notes"]


def load(path: Path) -> list[dict]:
    """Read a golden set .xlsx. Returns list of dicts with SCHEMA_COLUMNS keys.

    run:<experiment> columns are ignored.
    Rows with an empty question are skipped.
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl required: pip install openpyxl")

    wb = openpyxl.load_workbook(path)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]

    items = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if all(v is None for v in row):
            continue
        d = dict(zip(headers, row))
        question = d.get("question")
        if not question:
            continue
        items.append({
            "id": d.get("id"),
            "question": str(question).strip(),
            "reference_answer": str(d.get("reference_answer") or "").strip(),
            "expected_function": str(d.get("expected_function") or "").strip(),
            "human_notes": d.get("human_notes"),
        })
    return items


def save(items: list[dict], path: Path) -> None:
    """Write items to .xlsx with SCHEMA_COLUMNS column order.

    Creates parent directories if needed. Overwrites existing file.
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl required: pip install openpyxl")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(SCHEMA_COLUMNS)
    for item in items:
        ws.append([item.get(col) for col in SCHEMA_COLUMNS])
    wb.save(path)


def append_run_column(path: Path, experiment: str, results: dict) -> None:
    """Add or overwrite column run:<experiment> in existing .xlsx.

    Matches rows by the id column. results maps question id -> value string.
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl required: pip install openpyxl")

    path = Path(path)
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    col_name = f"run:{experiment}"

    if col_name in headers:
        col_idx = headers.index(col_name) + 1  # openpyxl is 1-based
    else:
        col_idx = len(headers) + 1
        ws.cell(row=1, column=col_idx, value=col_name)

    id_col_idx = (headers.index("id") + 1) if "id" in headers else None

    for row_idx in range(2, ws.max_row + 1):
        if id_col_idx is None:
            break
        row_id = ws.cell(row=row_idx, column=id_col_idx).value
        if row_id in results:
            ws.cell(row=row_idx, column=col_idx, value=results[row_id])

    wb.save(path)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_golden_set.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add evals/golden_set.py tests/test_golden_set.py
git commit -m "feat: add evals/golden_set.py — Excel-native golden set I/O"
```

---

## Task 2: Update `eval_chunking.py`

**Files:**
- Modify: `evals/eval_chunking.py`

- [ ] **Step 1: Write a failing test that verifies xlsx is accepted and run column is appended**

Add to `tests/test_golden_set.py`:

```python
def test_eval_chunking_accepts_xlsx_and_appends_run_column(tmp_path):
    """eval_chunking.load_golden_set is gone; xlsx round-trips through golden_set.load."""
    from evals.golden_set import save, load
    items = [
        {"id": 1, "question": "What is CSCE 611 about?", "reference_answer": "OS course.",
         "expected_function": "hybrid_course", "human_notes": None},
    ]
    path = tmp_path / "golden.xlsx"
    save(items, path)
    loaded = load(path)
    # confirm load_golden_set no longer exists in eval_chunking
    import evals.eval_chunking as ec
    assert not hasattr(ec, "load_golden_set")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_golden_set.py::test_eval_chunking_accepts_xlsx_and_appends_run_column -v
```

Expected: FAIL — `assert not hasattr(ec, 'load_golden_set')` fails (function still exists)

- [ ] **Step 3: Replace loader in `eval_chunking.py`**

At the top of `evals/eval_chunking.py`, add the import after the existing imports:

```python
from evals.golden_set import load as _load_golden_set, append_run_column as _append_run_column
```

Delete the entire `load_golden_set` function (lines ~38–46):

```python
# DELETE this function entirely:
def load_golden_set(path: Path) -> list[dict]:
    """Load a golden set JSONL file. Returns list of question dicts."""
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
```

In `main()`, replace the call:

```python
# OLD:
golden_items = load_golden_set(args.golden_set)
# NEW:
golden_items = _load_golden_set(args.golden_set)
```

Update the `--golden-set` argument help text:

```python
# OLD:
help="Path to golden set JSONL (e.g. tamu_data/evals/golden_sets/golden_*.jsonl)",
# NEW:
help="Path to golden set .xlsx (e.g. tamu_data/evals/golden_sets/golden_*.xlsx)",
```

Update the docstring at the top of the file:

```python
# OLD:
        --golden-set tamu_data/evals/golden_sets/golden_20260313_draft_v1.jsonl \\
# NEW:
        --golden-set tamu_data/evals/golden_sets/golden_20260411_v1.xlsx \\
```

- [ ] **Step 4: Capture chunk info and append run column after the eval loop**

In `_run_one_query`, the row dict contains `_chunks` (list of dicts with `course_id`, `category`, `score`). The loop in `run_eval` pops `_chunks` before appending to results. We need to capture the run column value before that pop, and store it on the row for later collection.

In `run_eval`, just **before** `row.pop("_chunks", None)` (around line 430), add:

```python
        # Build run column value: "CSCE 670 §SCHEDULE 0.87, CSCE 638 §LO 0.71"
        run_col_parts = []
        for c in row.get("_chunks", []):
            cid = c.get("course_id", "?")
            cat = c.get("category", "?")
            score = c.get("score")
            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
            run_col_parts.append(f"{cid} §{cat} {score_str}")
        row["_run_col_value"] = ", ".join(run_col_parts)
```

Then, still in `run_eval`, just before `return results, run_name`, add collection of the run column dict (keyed by golden item id):

```python
    # Build id → run_col_value map for append_run_column
    # We need to pair results back to golden item ids.
    # golden_items order matches results order (skipped items were `continue`d).
    # Build a question→id lookup from golden_items.
    question_to_id = {item.get("question", ""): item.get("id") for item in golden_items}
    run_col_results = {}
    for r in results:
        qid = question_to_id.get(r.get("query", ""))
        val = r.pop("_run_col_value", None)
        if qid is not None and val is not None:
            run_col_results[qid] = val

    return results, run_name, run_col_results
```

Update the `run_eval` return type annotation and the call in `main()`:

In `main()`, change:

```python
# OLD:
results, run_name = run_eval(
# NEW:
results, run_name, run_col_results = run_eval(
```

After `print_summary(results, run_name, aggregates)` in `main()`, add:

```python
    if run_col_results:
        _append_run_column(args.golden_set, run_name, run_col_results)
        print(f"Run column appended to {args.golden_set}  (run:{run_name})")
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
python -m pytest tests/test_golden_set.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add evals/eval_chunking.py tests/test_golden_set.py
git commit -m "feat: eval_chunking reads .xlsx and appends run column after eval"
```

---

## Task 3: Update `run_benchmark.py`

**Files:**
- Modify: `evals/run_benchmark.py`

- [ ] **Step 1: Replace JSONL loader**

At the top of `evals/run_benchmark.py`, add after existing imports:

```python
from evals.golden_set import load as _load_golden_set, append_run_column as _append_run_column
```

In `main()`, replace the manual JSONL loading block (lines ~669–681):

```python
# DELETE:
    items = []
    with golden_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

# REPLACE WITH:
    items = _load_golden_set(golden_path)
```

Update the `--golden-set` help text:

```python
# OLD:
help="Path to golden set JSONL"
# NEW:
help="Path to golden set .xlsx"
```

Update the docstring at the top of the file:

```python
# OLD:
        --golden-set tamu_data/evals/golden_sets/golden_20260311_v1.jsonl \
# NEW:
        --golden-set tamu_data/evals/golden_sets/golden_20260411_v1.xlsx \
```

- [ ] **Step 2: Handle missing stratum and source_course_id gracefully**

`run_benchmark.py` uses `item.get("stratum", "")` and `item.get("source_course_id", "")` — both fields are dropped from the new schema. They will return `""` via `.get()` defaults, which is already correct. No additional changes needed.

- [ ] **Step 3: Append run column after benchmark completes**

In `main()`, after `write_markdown(rows, args.experiment_name, md_path)`, add:

```python
    # Append pipeline answers as a run column in the golden set Excel
    question_to_id = {item.get("question", ""): item.get("id") for item in items}
    run_col_results = {}
    for row in rows:
        qid = question_to_id.get(row.question)
        if qid is not None and row.answer_full:
            run_col_results[qid] = row.answer_full
    if run_col_results:
        _append_run_column(golden_path, args.experiment_name, run_col_results)
        print(f"Run column appended to {golden_path}  (run:{args.experiment_name})")
```

- [ ] **Step 4: Smoke test — verify run_benchmark imports cleanly**

```bash
python -c "import evals.run_benchmark; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add evals/run_benchmark.py
git commit -m "feat: run_benchmark reads .xlsx and appends pipeline answers as run column"
```

---

## Task 4: Update `import_eval_draft.py`

**Files:**
- Modify: `evals/import_eval_draft.py`

- [ ] **Step 1: Replace JSONL output with Excel**

At the top of `evals/import_eval_draft.py`, add after existing imports:

```python
from evals.golden_set import save as _save_golden_set
```

Replace the entire "Save" block in `main()` (~lines 123–131):

```python
# DELETE:
    GOLDEN_SETS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d")
    out_path = GOLDEN_SETS_DIR / f"golden_{ts}_{args.tag}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for item in golden_items:
            f.write(json.dumps(item) + "\n")

# REPLACE WITH:
    GOLDEN_SETS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d")
    out_path = GOLDEN_SETS_DIR / f"golden_{ts}_{args.tag}.xlsx"
    _save_golden_set(golden_items, out_path)
```

Update the `golden_items` builder to use the new schema (drop removed fields, rename `human_judgment` → `human_notes`). Replace the item dict construction (~lines 104–121):

```python
# OLD:
        item = {
            "question": str(row.get("question", "")).strip(),
            "reference_answer": str(row.get("reference_answer", "")).strip(),
            "stratum": str(row.get("stratum", "")).strip(),
            "category": str(row.get("category", "")).strip() or None,
            "source_crn": str(row.get("source_crn", "")).strip() or None,
            "source_course_id": source_course_id,
            "source_category": str(row.get("category", "")).strip() or None,
            "expected_function": str(row.get("expected_function", "")).strip(),
            "expected_course_ids": [source_course_id] if source_course_id else [],
            "expected_specific_categories": [],
            "expected_semantic_intent": False,
            "human_judgment": row.get("human_judgment"),
        }

# NEW:
        item = {
            "id": row.get("id") or (len(golden_items) + 1),
            "question": str(row.get("question", "")).strip(),
            "reference_answer": str(row.get("reference_answer", "")).strip(),
            "expected_function": str(row.get("expected_function", "")).strip(),
            "human_notes": row.get("human_notes") or row.get("human_judgment"),
        }
```

Update the final print at the bottom:

```python
# OLD:
    print(f"\nNext: make bench GOLDEN={out_path} EXP=<experiment_name>")
# NEW:
    print(f"\nNext: python evals/eval_chunking.py --golden-set {out_path} --experiment <name>")
```

Remove unused `json` import from `import_eval_draft.py` if it's no longer used elsewhere in that file.

- [ ] **Step 2: Smoke test**

```bash
python -c "import evals.import_eval_draft; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add evals/import_eval_draft.py
git commit -m "feat: import_eval_draft writes .xlsx instead of JSONL"
```

---

## Task 5: Migration — convert JSONL → Excel, delete JSONL files

**Files:**
- Create: `evals/migrate_golden_sets.py` (run once, then delete)

- [ ] **Step 1: Write the migration script**

Create `evals/migrate_golden_sets.py`:

```python
"""One-off migration: convert JSONL golden sets → Excel, delete JSONL files.

Run from repo root:
    python evals/migrate_golden_sets.py

golden_20260411_v1.xlsx already exists (user-edited) — skip regenerating it,
just delete its JSONL counterpart.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.golden_set import save

GOLDEN_DIR = Path("tamu_data/evals/golden_sets")

# Files that already have a user-edited .xlsx — just delete the JSONL
SKIP_CONVERT = {"golden_20260411_v1", "golden_20260410_v2", "CSCE_25gradcourses_nonrecursive"}

# Throwaway subsets — delete only, no Excel needed
DELETE_ONLY = {"smoke_semantic_1q", "golden_20260411_v1_reworked"}


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def jsonl_to_golden_item(raw: dict, idx: int) -> dict:
    return {
        "id": raw.get("id") or idx,
        "question": str(raw.get("question", "")).strip(),
        "reference_answer": str(raw.get("reference_answer", "")).strip(),
        "expected_function": str(raw.get("expected_function", "")).strip(),
        "human_notes": raw.get("human_notes") or raw.get("human_judgment"),
    }


def main():
    jsonl_files = list(GOLDEN_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found.")
        return

    for jf in sorted(jsonl_files):
        stem = jf.stem

        if stem in DELETE_ONLY:
            jf.unlink()
            print(f"  deleted (throwaway): {jf.name}")
            continue

        if stem in SKIP_CONVERT:
            jf.unlink()
            print(f"  deleted (xlsx exists): {jf.name}")
            continue

        xlsx_path = jf.with_suffix(".xlsx")
        if xlsx_path.exists():
            jf.unlink()
            print(f"  deleted (xlsx exists): {jf.name}")
            continue

        # Convert
        raw_items = load_jsonl(jf)
        items = [jsonl_to_golden_item(r, i + 1) for i, r in enumerate(raw_items)]
        save(items, xlsx_path)
        jf.unlink()
        print(f"  converted: {jf.name} → {xlsx_path.name}  ({len(items)} rows)")

    print("\nMigration complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the migration**

```bash
python evals/migrate_golden_sets.py
```

Expected output (approximately):
```
  converted: golden_20260313_draft_v1.jsonl → golden_20260313_draft_v1.xlsx  (51 rows)
  converted: golden_20260313_draft_v1_sample10.jsonl → golden_20260313_draft_v1_sample10.xlsx  (10 rows)
  deleted (xlsx exists): golden_20260410_v2.jsonl
  deleted (xlsx exists): golden_20260411_v1.jsonl
  deleted (xlsx exists): CSCE_25gradcourses_nonrecursive.jsonl
  converted: smoke_v1.jsonl → smoke_v1.xlsx  (3 rows)
  deleted (throwaway): smoke_semantic_1q.jsonl
  deleted (throwaway): golden_20260411_v1_reworked.jsonl

Migration complete.
```

- [ ] **Step 3: Verify no JSONL files remain**

```bash
ls tamu_data/evals/golden_sets/
```

Expected: only `.xlsx` files visible.

- [ ] **Step 4: Commit migration script and results**

```bash
git add evals/migrate_golden_sets.py tamu_data/evals/golden_sets/
git commit -m "feat: migrate golden sets from JSONL to Excel, delete JSONL files"
```

- [ ] **Step 5: Delete the migration script (it's single-use)**

```bash
git rm evals/migrate_golden_sets.py
git commit -m "chore: remove one-off migration script"
```

---

## Task 6: Update `run-eval` skill

**Files:**
- Modify: `/home/claude/.claude/skills/run-eval/SKILL.md`

- [ ] **Step 1: Update the skill**

In `SKILL.md`, make two changes:

Change the discovery command:

```bash
# OLD:
ls tamu_data/evals/golden_sets/*.jsonl

# NEW:
ls tamu_data/evals/golden_sets/*.xlsx
```

Update the example golden set label in the confirmation block:

```
# OLD:
  Golden set:    [1] golden_20260313_draft_v1_sample10.jsonl (10 q)

# NEW:
  Golden set:    [1] golden_20260313_draft_v1_sample10.xlsx (10 q)
```

Update the `wc -l` note — Excel files don't have lines, so change to use python to count rows:

```bash
# The step 1 command becomes:
ls tamu_data/evals/golden_sets/*.xlsx
```

And for question counts, use:
```bash
python -c "
import openpyxl, glob
for f in sorted(glob.glob('tamu_data/evals/golden_sets/*.xlsx')):
    wb = openpyxl.load_workbook(f, read_only=True)
    n = wb.active.max_row - 1
    print(f'{n:3d}  {f}')
"
```

- [ ] **Step 2: Verify skill looks correct**

```bash
cat /home/claude/.claude/skills/run-eval/SKILL.md
```

- [ ] **Step 3: Commit**

```bash
git add /home/claude/.claude/skills/run-eval/SKILL.md
git commit -m "chore: update run-eval skill to use .xlsx golden sets"
```

---

## Self-Review

**Spec coverage:**
- `golden_set.py` with `load`, `save`, `append_run_column` → Task 1 ✓
- `eval_chunking.py` swap + run column → Task 2 ✓
- `run_benchmark.py` swap + run column → Task 3 ✓
- `import_eval_draft.py` writes xlsx → Task 4 ✓
- Migration converts + deletes JSONL → Task 5 ✓
- `run-eval` skill updated → Task 6 ✓
- `golden_20260411_v1.xlsx` not overwritten → Task 5 `SKIP_CONVERT` set ✓
- `human_judgment` → `human_notes` → Task 4 item builder + Task 5 migration both handle both field names ✓

**Type consistency:**
- `append_run_column(path, experiment, results: dict)` — called in Task 2 and 3 with `dict[int, str]` → consistent ✓
- `load()` returns `list[dict]` — Task 2 assigns to `golden_items`, Task 3 assigns to `items` → both iterate with `.get()` calls → consistent ✓
- `run_eval` now returns 3-tuple `(results, run_name, run_col_results)` — main() updated to unpack all 3 → consistent ✓
