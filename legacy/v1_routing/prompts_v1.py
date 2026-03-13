"""V1 generator prompts — preserved from the pre-v3-rework architecture.

The v1 system had one system prompt entry per routing function (8 total).
The v3 system uses 3 hybrid_course variants selected by specific_categories/specific_only,
plus a single recurrent prompt.
"""

# Primary prompt per function
FUNCTION_PROMPTS_V1: dict[str, str] = {
    "metadata_default": (
        "The user is asking for a general overview of a course. "
        "Provide key facts about course overview, prerequisites, and learning outcomes. "
        "Include the course ID and section. Label information by section where multiple sections are present."
    ),
    "metadata_specific": (
        "The user is asking about specific course details. "
        "Focus on the requested categories and be precise and complete. "
        "Include the course ID and section. Name the instructor where relevant."
    ),
    "metadata_combined": (
        "The user is asking about specific course details in the context of a broader overview. "
        "Cover both the requested categories and the general course overview. "
        "Include the course ID and section."
    ),
    "semantic_general": (
        "The user has a broad question not tied to a specific course. "
        "First define the relevant principle or framework underlying the question, "
        "then apply that principle to the specific question using available context. "
        "Provide a helpful answer based only on the available context. "
        "If the evidence is insufficient to answer fully, state: "
        "'I don't have enough data to answer this accurately based on the available syllabi.'"
    ),
    "recurrent_default": (
        "The user wants to find courses that pair well with or complement a specific course. "
        "The context includes the anchor course's overview and the most relevant discovered courses. "
        "Explain what makes each discovered course a good complement, grounding your reasoning "
        "in the anchor course's content (prerequisites, learning outcomes, covered topics)."
    ),
    "recurrent_specific": (
        "The user wants to find courses that complement a specific course, with focus on particular categories. "
        "Use the category-specific evidence from both the anchor course and the discovered courses "
        "to explain the pairing rationale. Ground all recommendations in the provided context."
    ),
    "recurrent_combined": (
        "The user wants course recommendations complementing a specific course, with both a "
        "broad overview and category-specific focus. Use all available evidence from the anchor "
        "and discovered courses to explain why each recommendation fits."
    ),
}

# V1 STRATUM_MAP — evaluation strata aligned to 8-function system
# (from evals/generate_golden_set.py before the v3 rework)
STRATUM_MAP_V1: dict[str, dict] = {
    "metadata_default": {
        "expected_function": "metadata_default",
        "n_questions": 10,
        "has_category": False,
        "expected_semantic_intent": False,
        "expected_recurrent_search": False,
    },
    "metadata_default_advisory": {
        "expected_function": "metadata_default",
        "n_questions": 10,
        "has_category": False,
        "expected_semantic_intent": True,
        "expected_recurrent_search": False,
    },
    "semantic_general": {
        "expected_function": "semantic_general",
        "n_questions": 8,
        "has_category": False,
        "expected_semantic_intent": True,
        "expected_recurrent_search": False,
    },
    "metadata_specific": {
        "expected_function": "metadata_specific",
        "n_questions": 10,
        "has_category": True,
        "expected_semantic_intent": False,
        "expected_recurrent_search": False,
    },
    "metadata_specific_evaluative": {
        "expected_function": "metadata_specific",
        "n_questions": 6,
        "has_category": True,
        "expected_semantic_intent": True,
        "expected_recurrent_search": False,
    },
    "metadata_combined": {
        "expected_function": "metadata_combined",
        "n_questions": 4,
        "has_category": True,
        "expected_semantic_intent": False,
        "expected_recurrent_search": False,
    },
    "recurrent_default": {
        "expected_function": "recurrent_default",
        "n_questions": 4,
        "has_category": False,
        "expected_semantic_intent": True,
        "expected_recurrent_search": True,
    },
    "out_of_scope": {
        "expected_function": "out_of_scope",
        "n_questions": 2,
        "has_category": False,
        "expected_semantic_intent": False,
        "expected_recurrent_search": False,
    },
}
# Total: 10+10+8+10+6+4+4+2 = 54, trimmed to 50
