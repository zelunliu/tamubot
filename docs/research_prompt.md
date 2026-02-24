# RAG Architecture Research Prompt

## What I'm Building

A chatbot (TamuBot) that answers questions about Texas A&M University academics for students.
Stack: Python, Streamlit frontend, Google Vertex AI RAG Engine (Managed Spanner Corpus), Gemini 2.5 Flash.

Example questions it needs to answer:
- "How do CSCE 638 and CSCE 670 overlap?"
- "What is the grading policy for ACCT 209?"
- "What are the prerequisites for MATH 151?"
- "Which professors teach AERO 214 this semester?"
- "What is the AI policy for courses in the College of Engineering?"
- "What degree requirements does the Department of Mathematics have?"

---

## Data I Have

### Source 1: University Catalog Pages
~100 Markdown files scraped from catalog.tamu.edu. Each file is a department or degree program page.

**Format:**
```markdown
---
url: https://catalog.tamu.edu/undergraduate/arts-and-sciences/mathematics/
title: "Department of Mathematics < Texas A&M Catalogs"
---

# Department of Mathematics

The Department of Mathematics offers curricula which lead to the following
undergraduate degrees: Bachelor of Science in Applied Mathematics, Bachelor
of Arts in Mathematics and Bachelor of Science in Mathematics...

Allen, Angela J, Instructional Associate Professor
Mathematics
MS, Texas A&M University, 2005
...
```

**Characteristics:**
- Long documents (faculty lists, program descriptions, degree requirements, course listings)
- Mostly unstructured prose + faculty directory entries
- No consistent section headings

---

### Source 2: Syllabus PDFs (Raw Extracted Text)
7,889 `.txt` files — one per course section for Spring 2026, extracted from PDFs using PyMuPDF.
Named: `{term}_{SUBJ}_{course}_{section}_{crn}.txt`

**Example (`202611_ACCT_229_200_10003.txt`):**
```
Mays Business School
Accounting
ACCT 229 Syllabus
Section 200 (10003)
Introductory Accounting
Spring 2026 - College Station

Course Information
Meeting Times: LEC  TR  10:05AM–11:20AM
Start Date: 01/12/2026  End Date: 05/05/2026
Meeting Location: WCBA 102
Credit Hours: 3

Instructor Details
Jacqueline Knoop
Email: jknoop@mays.tamu.edu
Office Location: Wehner 460Z
Phone: 979-845-9695
...
```

---

### Source 3: Standardized Syllabus Chunks (Current Pre-processed Form)
7,889 `.json` files. Each PDF was parsed with PyMuPDF to detect visually styled text (larger font/bold)
as section headers. Content between headers became chunks. Headers were mapped to 12 standard categories
via regex pattern matching.

**Scale:** 165,720 total chunks, avg 21 per file, range 0–215.

**Schema:**
```json
[
  {
    "course": "ACCT 209",
    "standard_header": "COURSE_INFO",
    "raw_header": "Introduction",
    "content": "Course Information – Spring 2026  Course Number: ACCT 209..."
  },
  {
    "course": "ACCT 209",
    "standard_header": "GRADING",
    "raw_header": "Grading Policy",
    "content": "Your course grade is determined based on total points. A: 360–400, B: 320–359..."
  },
  {
    "course": "ACCT 209",
    "standard_header": "MISC_OR_UNKNOWN",
    "raw_header": "Examination Schedule:",
    "content": "EXAM CHAPTERS LOCATION TIME DATE\nExam #1 Ch. 1,2,3,4 On CANVAS 7-8:15pm 2/17..."
  }
]
```

**Header distribution across all 165,720 chunks:**
| Category | Count | % |
|---|---|---|
| MISC_OR_UNKNOWN | 65,655 | 39.6% |
| UNIVERSITY_POLICIES | 25,561 | 15.4% |
| COURSE_INFO | 23,914 | 14.4% |
| SCHEDULE | 11,623 | 7.0% |
| LATE_WORK | 8,646 | 5.2% |
| GRADING | 7,766 | 4.7% |
| ATTENDANCE | 4,748 | 2.9% |
| MATERIALS | 4,232 | 2.6% |
| LEARNING_OUTCOMES | 3,749 | 2.3% |
| INSTRUCTOR_INFO | 3,263 | 2.0% |
| PREREQUISITES | 2,640 | 1.6% |
| SUPPORT_SERVICES | 1,962 | 1.2% |
| AI_POLICY | 1,961 | 1.2% |

**Key data quality issues:**
- 2,453 chunks have empty content (noise from PDF extraction)
- ~40% is MISC_OR_UNKNOWN (exam schedules, reading lists, course calendars — real content but
  didn't match any regex header pattern)
- UNIVERSITY_POLICIES (15.4%) is nearly identical boilerplate across every course (FERPA, ADA,
  Title IX, mental health, etc.) — copy-pasted by all instructors from a university template

---

## Current Problems / What I'm Unsure About

1. **Granularity**: Is one JSON file per course-section (21 chunks avg) the right upload unit?
   Or should I upload each chunk individually? Or group by category across courses?

2. **Boilerplate pollution**: 25,561 UNIVERSITY_POLICIES chunks are essentially the same text
   repeated 7,889 times across every course. This will dominate retrieval for policy questions
   and waste corpus space. Should I deduplicate?

3. **MISC_OR_UNKNOWN at 39.6%**: Nearly half the data has no semantic category. Should I
   try harder to classify it, discard it, or include it as-is?

4. **Metadata vs. text**: I want to filter by `standard_header` (e.g., "only look at GRADING
   chunks for this query") but the current system (Vertex AI RAG Engine / Spanner) embeds
   metadata in the text body as `[GRADING] raw_header\ncontent`. Is this the right approach
   or should metadata be stored separately for filtered retrieval?

5. **Catalog vs. Syllabi**: Should these be in the same vector index or separate indices?
   Catalog is "what does the program require?" — Syllabus is "what does this section of this
   course do this semester?" Different retrieval patterns.

6. **RAG system choice**: I'm using Vertex AI RAG Engine (managed Spanner). Is this the
   right tool for this use case or should I switch to something else (pgvector, Weaviate,
   ChromaDB, LlamaIndex, etc.)? I'd prefer to use an existing open-source GitHub project
   rather than building from scratch.

---

## My Questions

1. **What is the ideal chunk schema/format** for this data to maximize retrieval quality for
   complex multi-hop questions (e.g., "compare grading policies across CSCE courses")?

2. **How should I handle the boilerplate UNIVERSITY_POLICIES content?** Deduplicate to one
   canonical copy? Keep per-course for completeness? Filter entirely from index?

3. **Should I use a different RAG framework?** Is there a well-maintained open-source project
   on GitHub that handles university/document Q&A well out of the box that I could adapt?
   (LlamaIndex, Haystack, RAGFlow, anything else?)

4. **What metadata should be stored alongside each chunk** to enable useful filtering?
   (course code, department, section, term, category, instructor?)

5. **Should catalog and syllabus data be in the same corpus?** What are the trade-offs?

6. **What should I do about MISC_OR_UNKNOWN?** Is 39.6% unclassified content a problem for RAG?

Please give concrete recommendations, not just trade-offs. I'm a developer who wants to implement
the best practical solution without over-engineering it.
