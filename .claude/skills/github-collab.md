# github-collab — Branch + PR Workflow

Automate branch creation, pre-PR checks, pushing, and cleanup for team collaboration.

## Auto-Engage Triggers

Invoke this skill automatically (no `/` command needed) when the user:
- Says "push", "push this", "push my changes", "push to GitHub"
- Says "open a PR", "create a PR", "submit a PR", "make a pull request"
- Says "start a feature", "create a branch", "new branch for X"
- Says "clean up", "I merged", "branch is merged", "done with this branch"
- Asks "am I ready to push?", "can I open a PR?", "is this ready?"
- Says "check my branch", "run checks", "is my code clean?"

When auto-engaging, infer the most appropriate subcommand from context.

---

## Subcommands

### `start <feature-name>`

Create a new branch for a feature. Use when the user wants to start isolated work.

**Validation:**
1. `<feature-name>` must be provided and kebab-case → abort if missing or has spaces/uppercase
2. Check branch doesn't exist on origin:
   ```bash
   git fetch origin && git ls-remote --heads origin feature/<feature-name>
   ```
   Non-empty output → abort: "Branch already exists on origin."
3. Warn if main is behind:
   ```bash
   git log HEAD..origin/main --oneline
   ```
   Non-empty → warn: "Pull main first: `git pull origin main`"

**Create branch:**
```bash
git checkout -b feature/<feature-name>
```

Print: branch name + next step ("make your changes, then I'll run checks and push when ready").

---

### `check`

Pre-PR gate. Run all steps; stop on first failure.

**Step 1 — Rebase check:**
```bash
git fetch origin && git log HEAD..origin/main --oneline
```
Non-empty → FAIL: "Behind origin/main. Run: `git rebase origin/main`"

**Step 2 — Lint:**
```bash
make lint
```

**Step 3 — Typecheck:**
```bash
make typecheck
```

**Step 4 — Tests:**
```bash
make test
```

**Step 5 — Conflict markers:**
```bash
grep -r "<<<<<<< HEAD" --include="*.py" --include="*.md" --include="*.json" -l .
```
Any hits → FAIL: "Conflict markers in: <files>"

**Step 6 — Golden set warning:**
```bash
git diff --name-only origin/main...HEAD -- tamu_data/evals/golden_sets/
```
Any files → WARN: "Golden set modified — coordinate with teammates before merging (JSON has no semantic merge)."

**Print summary table:**
```
Rebase:    PASS / FAIL
Lint:      PASS / FAIL
Typecheck: PASS / FAIL
Tests:     PASS / FAIL
Conflicts: PASS / FAIL
```
All PASS → "Ready. Run checks again or tell me to push." Any FAIL → "Fix above, then I'll re-run."

---

### `pr`

Run `check`, then push and give the PR link. This is the primary auto-engage target for "push" / "open PR" intent.

1. Run all `check` steps — abort if any fail, print what to fix
2. Confirm: "Push `<branch>` and open a PR? (yes/no)"
3. Push:
   ```bash
   git push -u origin <current-branch>
   ```
4. Print PR URL:
   ```
   https://github.com/artemkorolev1/tamubot/compare/<branch>?expand=1
   ```

---

### `finish`

Post-merge cleanup. Auto-engage when user says the branch is merged.

1. Detect current branch:
   ```bash
   git branch --show-current
   ```
2. Check if still on origin:
   ```bash
   git fetch origin && git ls-remote --heads origin <branch>
   ```
   Still exists → WARN: "Branch still on origin — may not be merged yet. Continue? (yes/no)"
3. Confirm: "Delete branch `<branch>` and pull main? (yes/no)"
4. Cleanup:
   ```bash
   git checkout main
   git branch -d <branch>
   git pull origin main
   ```

---

### `status`

Show current branch state at a glance.

```bash
git branch --show-current
git status --short
git log origin/main..HEAD --oneline
git log HEAD..origin/main --oneline
```

Print:
```
Branch:   feature/rag-reranker
Changes:  <git status --short or "none">
Ahead:    <commits ahead or "none">
Behind:   <commits behind or "up to date">
```

---

## High-Conflict Files (warn when modified)

- `rag/prompts.py` — assign ownership per sprint
- `tamu_data/evals/golden_sets/*.json` — one owner per sprint; coordinate before merge
- `requirements.txt` — always rebase before opening PR
- `CLAUDE.md` — resolve manually, keep union of both changes

## Reference

- Repo: `https://github.com/artemkorolev1/tamubot`
- Branch naming: `feature/<kebab>`, `fix/<kebab>`, `eval/<kebab>`
- Rebase, never merge: `git rebase origin/main` (not `git merge main`)
