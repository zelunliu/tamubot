# Research Prompts for TamuBot RAG Pipeline

Use each prompt as a separate Gemini Deep Research session.

---

## PROMPT 1: Chunking & Preprocessing Pipeline

(COMPLETED — we chose Gemini 2.5 Flash direct PDF parsing. See test_gemini_parse.py for implementation.)

**Summary of decisions made:**
- Gemini 2.5 Flash reads PDFs natively (multimodal) and outputs structured JSON in a single API call
- 11 semantic categories: COURSE_OVERVIEW, INSTRUCTOR, PREREQUISITES, LEARNING_OUTCOMES, MATERIALS, GRADING, SCHEDULE, ATTENDANCE_AND_MAKEUP, AI_POLICY, UNIVERSITY_POLICIES, SUPPORT_SERVICES
- Boilerplate university policies: only policy NAMES are extracted per course; full text stored once as a "golden copy"
- Completeness checking: Gemini flags missing sections and data quality warnings per syllabus
- Tables preserved as Markdown format
- 1 API call per PDF, ~20-60 seconds each, free tier
- Tested on CSCE 221, CSCE 331, CSCE 481, CSCE 482, ISEN 210, ISEN 302

---

## PROMPT 2: Database, Embedding & Retrieval

(COMPLETED — we chose MongoDB Atlas + Voyage AI. See Prompt 4 results for details.)

---

## PROMPT 3: LLM Orchestration Architecture for RAG

I'm building a RAG chatbot (TamuBot) for Texas A&M CS students. I need to design the **full LLM orchestration layer** — every LLM call in the pipeline, from user input to final response. I want a fundamental, unbiased exploration of best practices for my specific use case before committing to an architecture.

### System context

- **Frontend**: Streamlit chat UI
- **LLM**: Gemini 2.5 Flash (fast, cheap, structured JSON output)
- **Database**: MongoDB Atlas with 3 collections:
  - `chunks` (~3K docs): denormalized syllabus chunks, each with `course_id`, `section`, `term`, `instructor_name`, `category` (one of 11 types: GRADING, SCHEDULE, AI_POLICY, PREREQUISITES, LEARNING_OUTCOMES, etc.), `content`, `embedding` (Voyage AI voyage-3, 1024 dims)
  - `policies`: deduplicated university boilerplate policies (Academic Integrity, ADA, FERPA, etc.)
  - `courses`: one doc per section with full metadata, for aggregate queries
- **Search functions available**: `hybrid_search` (RRF fusion of vector + BM25), `search_semantic` (pure vector), `search_by_course` (metadata filter), `get_policy` (policy lookup), `aggregate_query` (count/comparison)
- **Scale**: 259 course sections now (CSCE + ISEN), expanding to all departments (~7,889 sections)

### What I need researched

**Part A — How many LLM calls do we actually need?**

Don't assume it's exactly two. Explore the full spectrum:
1. Single LLM call (stuff context + generate)
2. Two calls: router/classifier → retrieval → generator/synthesizer
3. Three+ calls: router → retrieval → re-ranker/filter → generator
4. Agentic loops (LangGraph, Google ADK, tool-calling agents)

For each pattern: when is it appropriate, what are the latency/cost/accuracy tradeoffs, and at what scale does each make sense? What's the right choice for ~3K chunks growing to ~50K?

**Part B — Inlet LLM (pre-retrieval)**

Whatever sits between the user's raw query and the database:
1. **Query classification**: How to distinguish queries that need retrieval (e.g. "What's the grading policy for CSCE 120?") from conversational queries that don't (e.g. "Howdy", "Thanks", "What can you help me with?")?
2. **Intent taxonomy**: What intents should we support? Consider: single-course lookup, cross-course comparison, aggregation/counting, policy lookup, instructor queries, schedule queries, conversational/chitchat, out-of-scope. Is this over-engineered or missing things?
3. **Entity extraction**: Pulling out course_id, section, category, instructor name, term from natural language. How to handle ambiguity ("the 400-level AI courses")?
4. **Retrieval strategy mapping**: How does each intent map to which search function? Should the router output a specific search function name, or just intent + entities?
5. **Query rewriting/expansion**: Should the inlet LLM rewrite vague queries before embedding? (e.g. "late work" → "late work policy attendance makeup")
6. **Safety/guardrails**: Handling off-topic queries, prompt injection attempts, and queries outside our knowledge domain gracefully
7. **Confidence and fallback**: When the router isn't sure, what's the fallback? Always run vector search? Ask the user to clarify?

**Part C — Outlet LLM (post-retrieval)**

Whatever takes retrieved documents and produces the final answer:
1. **Context formatting**: How to structure retrieved chunks in the prompt? Just concatenate? Include metadata headers? Numbered sources?
2. **Context window management**: With 5-10 retrieved chunks, how to fit them efficiently? Should we pre-filter/truncate before passing to the LLM?
3. **Grounding and hallucination prevention**: Best prompt patterns to keep the LLM strictly grounded in retrieved context. How to handle contradictions across different sections' syllabi.
4. **Citation/attribution**: How to make the LLM cite which course/section/category each fact came from, so the user can verify. Inline citations vs. footnotes vs. source list.
5. **No-results handling**: When retrieval returns nothing or low-relevance results, how should the outlet LLM respond? Admit it doesn't know vs. suggest related queries?
6. **Should it receive the router's classification?** Does passing the intent/entities to the generator help it produce better answers?
7. **Response format**: Should the outlet LLM be told the user's likely goal (e.g. "the user wants to compare grading policies") to structure its response appropriately?

**Part D — Practical considerations**

1. **Latency budget**: Users expect <3s for a chatbot response. If Gemini Flash is ~0.5-1s per call, how many LLM calls can we afford?
2. **Cost at scale**: With ~100 daily active users, what's the cost profile for each architecture?
3. **Caching**: Which LLM calls are cacheable? (e.g. identical course lookups)
4. **Streaming**: Can we stream the final response while still using a router? How does that work architecturally?
5. **Evaluation**: How to measure if our orchestration is actually working well? What metrics matter (retrieval precision, answer faithfulness, latency)?

### Example queries to consider in your analysis

**Single-course lookups:**
- "What's the grading policy for CSCE 120?" (specific category)
- "Who teaches CSCE 120 section 500?" (direct metadata lookup)
- "Can I use ChatGPT for homework in CSCE 120?" (specific policy within a course)
- "What are the prerequisites for CSCE 120?" (specific category)

**Cross-course comparison & course selection:**
- "Compare AI policies across all 400-level CSCE courses" (filtered comparison)
- "Which course should I take to learn about RAG systems and information retrieval?" (topic-based course recommendation — needs to search learning outcomes, course overviews, schedules across courses)
- "How much do CSCE 638 and CSCE 670 overlap in terms of topics?" (pairwise course comparison — retrieve both syllabi, compare learning outcomes and schedules)
- "Which CSCE courses have open-book finals?" (cross-course scan of grading/schedule)
- "Can I take CSCE 120 section 500 and CSCE 111 section 501 together? Do the schedules conflict?" (schedule conflict detection — compare meeting times between two specific sections)
- "I want to take a course on machine learning and one on databases next semester. Which sections don't conflict?" (multi-course schedule compatibility)

**Aggregation:**
- "How many sections of CSCE 120 are there?" (count query)

**Ambiguous queries:**
- "Tell me about the late work policy" (which course? or university-wide?)

**Conversational (no retrieval needed):**
- "Howdy!" (greeting — just respond friendly, don't hit the database)
- "Thanks, that was helpful!" (acknowledgment)
- "What can you help me with?" (capabilities question — describe what TamuBot can do)

**In-scope but insufficient data (partial knowledge):**
- "What's the average GPA for CSCE 120?" (related to course selection at TAMU, but we don't have GPA data in our syllabi — bot should acknowledge this is a valid question, explain we only have syllabus data, and suggest where to find GPA info like TAMU grade distribution reports)
- "Is CSCE 120 harder than CSCE 111?" (difficulty comparison — we have syllabi but not difficulty ratings; bot can compare workload from grading breakdowns but should caveat)
- "When is the registration deadline?" (TAMU academic question but not in our syllabus data — should redirect to registrar)
- "Does Professor Beideman curve grades?" (about a TAMU instructor we have data on, but curving info may not be in the syllabus)

**Fully out of scope (should be declined gracefully):**
- "What's the meaning of life?" (completely unrelated)
- "Write me a Python script to sort a list" (coding help, not course advising)
- "What's the weather in College Station?" (not academic)
- For these: the bot should politely decline with something like "I'm TamuBot, an academic course assistant for Texas A&M. I can help with course information, syllabi, scheduling, and policies. For other questions, try [general-purpose chatbot]. Let me know if I can help with course selection or planning!"

### Output format

Give me:
1. **Recommended architecture** with justification — how many LLM calls, what each does, why
2. **Detailed prompt templates** for each LLM call (I'll adapt these into code)
3. **Intent taxonomy** with mapping to retrieval strategies
4. **Edge case handling** — how each example query flows through the system
5. **Tradeoff analysis** — what we gain/lose with this architecture vs. simpler/more complex alternatives
6. **2-3 reference implementations** (GitHub repos, blog posts, papers) that follow similar patterns
7. **Implementation priority** — what to build first, what can be added later

---

## PROMPT 4: MongoDB Atlas Integration

(COMPLETED — research done. Summary below.)

**Decisions made:**
- MongoDB Atlas M0 (free tier), 3 collections: chunks (denormalized), policies (deduplicated), courses (metadata)
- Embedding: Voyage AI voyage-3 (1024 dims), with contextual anchors prepended before embedding
- Hybrid search: $vectorSearch + $search with $rankFusion (RRF), MongoDB 8.0
- Ingestion: idempotent upserts, SHA-256 policy deduplication, batch embeddings (50/batch)
- Indexes: vector_index (vectorSearch), text_index (search), compound metadata index
- Scale: works on M0 for 259 sections, scales to 7,889+ without re-architecture

**Key references:**
- https://github.com/mongodb/chatbot
- https://github.com/mongodb-industry-solutions/manufacturing-car-manual-RAG
- https://github.com/mongodb-developer/Google-Cloud-RAG-Langchain
- https://github.com/mongodb-industry-solutions/document-intelligence
