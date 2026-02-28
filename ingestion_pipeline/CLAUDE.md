# ingestion_pipeline/

## Gotcha

**Always uses `GOOGLE_API_KEY` directly** — not the TAMU gateway. PDF bytes (`Part.from_bytes`) are not supported by OpenAI-compatible APIs.

## Running

```bash
GOOGLE_API_KEY=... python ingestion_pipeline/process_syllabi.py [--department CSCE] [--retry-errors]
```

Resume: skips files in `tamu_data/logs/progress.jsonl`. Restart from scratch: delete that file.
Errors logged to `tamu_data/logs/errors.jsonl`. Output: `tamu_data/processed/gemini_parsed/<CRN>.json`.

## Output Schema

```json
{
  "course_metadata": { "course_id", "section", "term", "crn", "instructor", "tas", "meeting_times", "location" },
  "chunks": [{"category": "GRADING", "title": "...", "content": "...", "has_table": false}],
  "boilerplate_policies": [...],
  "completeness_check": {"missing_sections": ["SCHEDULE"], "warnings": [...]}
}
```

## 11 Categories

`COURSE_OVERVIEW` · `INSTRUCTOR` · `PREREQUISITES` · `LEARNING_OUTCOMES` · `MATERIALS` · `GRADING` · `SCHEDULE` · `ATTENDANCE_AND_MAKEUP` · `AI_POLICY` · `UNIVERSITY_POLICIES` · `SUPPORT_SERVICES`
