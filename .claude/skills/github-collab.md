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

## GitHub concepts — quick reference (explain when relevant)

Use these plain-English explanations whenever the user seems confused by a term:

- **branch**: a separate copy of the code you can edit freely without affecting `main`. Like a draft.
- **main**: the "official" version of the code everyone shares. Only merge tested, working changes here.
- **commit**: a saved snapshot of your changes, like hitting Save with a label attached.
- **push**: uploading your local commits to GitHub so others can see them.
- **PR (Pull Request)**: a request to merge your branch into main. GitHub shows a diff, lets teammates review, then merges it.
- **rebase**: replay your commits on top of the latest main, so your branch stays current. Preferred over "merge main in" because it keeps history clean.
- **conflict marker**: `<<<<<<< HEAD` in a file means git couldn't auto-merge two versions — you have to pick which one to keep manually.
- **lint**: automated style + error checker. Catches things like unused imports, undefined variables, bad formatting.
- **typecheck**: verifies that function arguments match their expected types (mypy). Catches bugs before runtime.

---

## Subcommands

### `start <feature-name>`

**What this does:** Creates a new branch so your work is isolated from `main`. You can experiment freely — nothing you do here affects the shared codebase until you open a PR and it gets merged.

**Validation:**
1. `<feature-name>` must be provided and kebab-case (lowercase, hyphens only) → abort if missing or has spaces/uppercase.
   - *Why kebab-case: branch names appear in URLs and terminal output; spaces and uppercase cause problems across tools.*
2. Check branch doesn't exist on origin:
   ```bash
   git fetch origin && git ls-remote --heads origin feature/<feature-name>
   ```
   Non-empty output → abort: "Branch already exists on GitHub. Use a different name or check it out with `git checkout feature/<feature-name>`."
   - *Why: duplicate branches cause confusion about which is the "real" one.*
3. Warn if main is behind:
   ```bash
   git log HEAD..origin/main --oneline
   ```
   Non-empty → warn: "Your local main is behind GitHub. Pull first: `git pull origin main`"
   - *Why: branching from stale main means your feature starts from old code — you'll have more conflicts later.*

**Create branch:**
```bash
git checkout -b feature/<feature-name>
```
*This creates the branch locally and switches to it. It doesn't exist on GitHub yet — that happens when you push.*

Print: branch name + "Make your changes. When ready, say 'push' or 'open a PR' and I'll run checks automatically."

---

### `check`

**What this does:** Runs a series of automated quality gates before allowing a push. Think of it as a preflight checklist — better to catch problems here than have a teammate review broken code.

Run each step in order. Stop and report on first failure.

**Step 1 — Rebase check**
```bash
git fetch origin && git log HEAD..origin/main --oneline
```
- Non-empty → FAIL
- *What this means: someone else pushed to main while you were working. Your branch is now "behind" — it doesn't have their changes. You need to rebase first so your changes sit on top of theirs, not beside them.*
- Fix: `git rebase origin/main` — then re-run check.

**Step 2 — Lint (ruff)**
```bash
python -m ruff check .
```
- Non-zero exit → FAIL, show errors
- *What this does: scans Python files for style errors, unused imports, undefined names, and common bugs. Fast — runs in under a second. Configured in `pyproject.toml`: line length 100, checks E (pycodestyle errors), F (pyflakes), I (import order).*
- Fix: `python -m ruff check . --fix` auto-corrects most issues.

**Step 3 — Typecheck (mypy)**
```bash
python -m mypy . --ignore-missing-imports
```
- Non-zero exit → FAIL, show errors
- *What this does: checks that function arguments and return types match their annotations. Catches bugs like passing a string where a list is expected — without running the code.*

**Step 4 — Tests (pytest)**
```bash
python -m pytest tests/ -v
```
- Non-zero exit → FAIL, show failures
- *What this does: runs the automated test suite. If existing tests break, your change likely broke something — even if the code looks fine.*

**Step 5 — Conflict markers**
```bash
grep -r "<<<<<<< HEAD" --include="*.py" --include="*.md" --include="*.json" -l . --exclude-dir=.venv --exclude-dir=.claude
```
- Any hits → FAIL: "Conflict markers found in: <files>"
- *What this means: a previous rebase or merge left unresolved conflicts in the file. You need to open those files, pick which version to keep, and delete the `<<<<<<<`, `=======`, `>>>>>>>` lines.*

**Step 6 — Golden set warning**
```bash
git diff --name-only origin/main...HEAD -- tamu_data/evals/golden_sets/
```
- Any files listed → WARN (not a failure): "Golden set files modified: <files>. Heads up — JSON has no semantic merge, so coordinate with teammates before this PR merges to avoid overwriting each other's labels."

**Print summary:**
```
Rebase:    PASS / FAIL
Lint:      PASS / FAIL
Typecheck: PASS / FAIL
Tests:     PASS / FAIL
Conflicts: PASS / FAIL
```
All PASS → "All checks passed. Say 'push' or 'open a PR' and I'll handle it."
Any FAIL → explain what failed and what to do to fix it.

---

### `pr`

**What this does:** Runs all checks, uploads your branch to GitHub, and gives you the link to open a PR. A PR (Pull Request) is how you ask for your branch to be merged into main — GitHub shows what changed, lets teammates leave comments, and records the merge.

1. Run all `check` steps — if any fail, stop and explain what to fix. Do not push.
2. Confirm with user: "All checks passed. Push branch `<branch>` to GitHub and open a PR? (yes/no)"
   - *Always confirm before pushing — pushing is visible to the whole team.*
3. Push:
   ```bash
   git push -u origin <current-branch>
   ```
   *`-u` sets the upstream so future `git push` commands don't need the full branch name.*
4. Print PR URL:
   ```
   https://github.com/artemkorolev1/tamubot/compare/<branch>?expand=1
   ```
   Tell the user: "Open that link, fill in a description of what you changed and why, then click 'Create pull request'. A teammate can review and merge it, or you can merge it yourself if you're working alone."

---

### `finish`

**What this does:** Cleans up after a PR is merged. Deletes the local branch and updates your local main to match GitHub. Skipping this leaves stale branches piling up.

1. Detect current branch:
   ```bash
   git branch --show-current
   ```
2. Check if still on GitHub:
   ```bash
   git fetch origin && git ls-remote --heads origin <branch>
   ```
   Still exists → WARN: "This branch still exists on GitHub — it may not be merged yet. Are you sure you want to clean up? (yes/no)"
   - *A merged PR usually deletes the branch on GitHub automatically. If it's still there, the PR might not be merged.*
3. Confirm: "Delete local branch `<branch>` and pull the latest main? (yes/no)"
4. Run cleanup:
   ```bash
   git checkout main
   git branch -d <branch>
   git pull origin main
   ```
   *`-d` (lowercase) is safe — it refuses to delete a branch that hasn't been merged. If you're sure it was merged and git still refuses, you can use `-D` (uppercase) to force it.*

Print: "Done. Branch deleted. Your local main is now up to date with GitHub."

---

### `status`

**What this does:** Quick snapshot — where are you, what's changed, how far ahead/behind is your branch vs main.

```bash
git branch --show-current   # which branch you're on
git status --short           # uncommitted file changes
git log origin/main..HEAD --oneline   # your commits not yet on main
git log HEAD..origin/main --oneline   # main commits you don't have yet (rebase lag)
```

Print clearly:
```
Branch:        feature/rag-reranker
               (your working branch — changes here don't affect main until merged)

Uncommitted:   app.py, rag/prompts.py
               (these changes exist only on your machine — commit them to save)

Ahead of main: 3 commits
               (commits you've made that aren't on main yet — will go into your PR)

Behind main:   0 commits
               (you're up to date — no rebase needed)
```

If behind > 0, explain: "Someone pushed to main while you were working. Run `git rebase origin/main` to pull their changes into your branch before opening a PR."

---

## High-conflict files — warn when modified

When any of these appear in `git diff --name-only origin/main...HEAD`, add a coordination note:

| File | Why it conflicts | What to do |
|------|-----------------|-----------|
| `rag/prompts.py` | Everyone tweaks prompts | Assign one person per sprint to own prompt changes |
| `tamu_data/evals/golden_sets/*.json` | JSON has no line-level semantic meaning — one person's edit overwrites another's | One owner per sprint; rebase before touching |
| `requirements.txt` | Two people adding different packages → duplicate/conflicting lines | Always rebase on main before opening PR |
| `CLAUDE.md` | Team adds gotchas from different angles | Resolve manually; keep the union of both changes |

---

## Reference

- Repo: `https://github.com/artemkorolev1/tamubot`
- Branch naming: `feature/<kebab>`, `fix/<kebab>`, `eval/<kebab>`
- Lint tool: `python -m ruff check .` (config in `pyproject.toml` — E, F, I rules, line length 100)
- Auto-fix lint: `python -m ruff check . --fix`
- Rebase, never merge: `git rebase origin/main` (not `git merge main`)
