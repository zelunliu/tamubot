"""Real end-to-end coreference test — uses actual LLM, embedder, reranker.

Validates routing AND the generated answer text for each turn.

Run inside the project root:
  python tools/test_coreference.py

Makes ~6 TAMU API calls and ~6 Voyage AI calls.
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from rag.graph.pipeline import run_pipeline_with_memory
from rag.graph.session import SessionManager
from rag.generator import generate_answer

session_manager = SessionManager()
session_id = f"coreference-test-{uuid.uuid4().hex[:8]}"
thread_config = session_manager.get_thread_config(session_id)

QUERIES = [
    "tell me about csce 638",
    "compare it with csce 670",
    "which one has stricter attendance policy",
    "how is csce 665 different from those?",
]

print("=" * 70)
print("COREFERENCE TEST — router + generated answer")
print(f"session: {session_id}")
print("=" * 70)

results = []
for i, query in enumerate(QUERIES, 1):
    print(f"\n{'─'*70}")
    print(f"TURN {i}: {query!r}")
    print("─" * 70)

    # Returns: (chunks, router_result, data_gaps, data_integrity, conflicted_course_ids, answer_tokens)
    chunks, rr, gaps, integrity, conflicts, answer_tokens = run_pipeline_with_memory(
        query, thread_config=thread_config
    )

    print(f"Router  → function={rr.function if rr else 'N/A'}  "
          f"course_ids={rr.course_ids if rr else 'N/A'}")
    print(f"Chunks  → {len(chunks)} retrieved")

    # Get the answer from answer_tokens
    answer = answer_tokens[0] if answer_tokens else ""

    print(f"\nAnswer:\n{answer[:800]}{'...' if len(answer) > 800 else ''}")
    results.append((query, rr, chunks, answer))

# ── Verdict ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("VERDICT")
_, rr1, _, _    = results[0]
_, rr2, _, ans2 = results[1]
_, rr3, _, ans3 = results[2]
_, rr4, _, ans4 = results[3]

t2_courses_ok = rr2 and "CSCE 638" in rr2.course_ids and "CSCE 670" in rr2.course_ids
t3_courses_ok = rr3 and len(rr3.course_ids) >= 1
t4_courses_ok = rr4 and "CSCE 665" in rr4.course_ids and len(rr4.course_ids) >= 2
t2_answer_ok  = "638" in ans2 and "670" in ans2
t3_answer_ok  = len(ans3) > 100
t4_answer_ok  = "665" in ans4 and len(ans4) > 100

print(f"  Turn 2 router has both courses:         {'PASS ✓' if t2_courses_ok else 'FAIL ✗'}  {rr2.course_ids if rr2 else 'N/A'}")
print(f"  Turn 2 answer mentions both:            {'PASS ✓' if t2_answer_ok  else 'FAIL ✗'}")
print(f"  Turn 3 router carries prior courses:    {'PASS ✓' if t3_courses_ok else 'FAIL ✗'}  {rr3.course_ids if rr3 else 'N/A'}")
print(f"  Turn 3 answer non-empty:                {'PASS ✓' if t3_answer_ok  else 'FAIL ✗'}")
print(f"  Turn 4 router adds CSCE 665 + retains:  {'PASS ✓' if t4_courses_ok else 'FAIL ✗'}  {rr4.course_ids if rr4 else 'N/A'}")
print(f"  Turn 4 answer mentions CSCE 665:        {'PASS ✓' if t4_answer_ok  else 'FAIL ✗'}")
