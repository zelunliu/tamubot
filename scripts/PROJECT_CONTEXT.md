# Project Context & Progress

## Recent Updates
- **[Syllabus Processing]** Finalized semantic chunking and header standardization.
    - **Method:** Visual header detection (PyMuPDF) + Pattern-based mapping.
    - **Efficiency:** 60.38% of 165,720 chunks mapped to standard headers (e.g., GRADING, AI_POLICY).
    - **Output:** Stored in `rag_vertex/standardized_chunks/`.
    - **Anomalies:** Logged in `final_standardization_report.json`.
- **[Data Quality]** Analyzed departmental standardization efficiency.
    - **Top Performers:** ASTR (97.9%), MSTC (91.8%), AGSM (87.1%).
    - **Chunk Characteristics:** Avg length varies from 600 to 2000 chars; high standard deviation indicates high content variance within sections.
- **[Data Distribution]** Analyzed chunk metrics by standard category.
    - **Largest Category:** UNIVERSITY_POLICIES (15.4% of content).
    - **Longest Chunks:** COURSE_INFO (Avg 1,886 chars, high variance).
    - **Most Consistent:** INSTRUCTOR_INFO and PREREQUISITES (Short, low-noise).
    - **RAG Implication:** High variance in COURSE_INFO and SCHEDULE suggests a need for secondary character-based splitting within those specific buckets.
- **[Configuration]** Operational guardrails (Plan First, Call Limits) active.
- **[Operational Rules]** 
    1. **Plan First:** Always provide a short plan before acting.
    2. **Call Limit:** Monitor tool call count (limit ~15 per task) and report if exceeded.
    3. **Reporting:** Provide a detailed report of actions and update this file after significant changes.
