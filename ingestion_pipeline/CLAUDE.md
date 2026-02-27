# ingestion_pipeline/ — Syllabus Data Pipeline

> **Maintenance**: Update this file when the parser schema, categories, or run commands change.

## Overview

`process_syllabi.py` is the production PDF parser. It sends each syllabus PDF directly to
Gemini 2.5 Flash (multimodal) and receives structured JSON back in a single API call.

**Always uses `GOOGLE_API_KEY` directly** — not the TAMU AI gateway. PDF bytes (`Part.from_bytes`)
are not supported by OpenAI-compatible APIs.

## Running

```bash
# Always run from the repo root
GOOGLE_API_KEY=... python ingestion_pipeline/process_syllabi.py

# Options
python ingestion_pipeline/process_syllabi.py --department CSCE    # single department
python ingestion_pipeline/process_syllabi.py --retry-errors       # retry previously failed files
```

## Resume Behavior

The pipeline **resumes automatically** from the last position:
- Reads `tamu_data/logs/progress.jsonl` — skips files already marked complete
- Logs errors to `tamu_data/logs/errors.jsonl`
- To restart from scratch: delete `tamu_data/logs/progress.jsonl`

## Output Schema (per PDF)

```json
{
  "course_metadata": {
    "course_id": "CSCE 638",
    "section": "501",
    "term": "Spring 2026",
    "crn": "12345",
    "instructor": {"name": "...", "email": "..."},
    "tas": [...],
    "meeting_times": [...],
    "location": "..."
  },
  "chunks": [
    {"category": "GRADING", "title": "...", "content": "...", "has_table": false}
  ],
  "boilerplate_policies": ["Academic Integrity Policy", "..."],
  "completeness_check": {
    "missing_sections": ["SCHEDULE"],
    "warnings": [...]
  }
}
```

## 11 Semantic Categories

| Category | Description |
|----------|-------------|
| `COURSE_OVERVIEW` | Course description, objectives, general scope |
| `INSTRUCTOR` | Instructor name, contact info, office hours |
| `PREREQUISITES` | Required/recommended prerequisite courses |
| `LEARNING_OUTCOMES` | What students will know/be able to do |
| `MATERIALS` | Required textbooks, software, tools |
| `GRADING` | Grade weights, breakdown, scale |
| `SCHEDULE` | Week-by-week topics, assignment due dates |
| `ATTENDANCE_AND_MAKEUP` | Attendance policy, late/makeup work rules |
| `AI_POLICY` | Rules for generative AI tool use |
| `UNIVERSITY_POLICIES` | Standard TAMU boilerplate policies |
| `SUPPORT_SERVICES` | Disability services, counseling, tutoring |

## Rate Limits

Built-in delays between API calls are included in the script.
If you hit quota errors, the `--retry-errors` flag will retry failed files on the next run.

## Output Location

Parsed JSONs written to: `tamu_data/processed/gemini_parsed/<CRN>.json`
