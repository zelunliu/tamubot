"""Context assembly for the generator stage.

Provides format_context_xml() which converts reranked retrieval results into
an XML-tagged context string with primacy-recency bracketing to combat
Lost-in-the-Middle attention degradation.
"""

import html
import re


def format_context_xml(results: list[dict]) -> str:
    """Format retrieval results as XML-tagged chunks for the generator.

    Implements primacy-recency bracketing to combat Lost-in-the-Middle attention degradation:
    - Rank 1 chunk → Context start (primacy position)
    - Rank 2 chunk → Context end (recency/nearest query position)
    - Ranks 3–N → Middle (descending rank order)

    Each chunk gets metadata attributes so the LLM can cite sources precisely.
    """
    if not results:
        return "<context>\nNo relevant documents found.\n</context>"

    # Apply primacy-recency reordering: [rank_1, ranks_3_to_N, rank_2]
    if len(results) == 1:
        ordered_results = results
        rank_mapping = [1]
    elif len(results) == 2:
        ordered_results = [results[0], results[1]]
        rank_mapping = [1, 2]
    else:
        # Rank 1 at start, Rank 2 at end, Ranks 3-N in middle (descending order)
        ordered_results = [results[0]] + results[2:] + [results[1]]
        rank_mapping = [1] + list(range(3, len(results) + 1)) + [2]

    parts = ["<context>"]
    for position, (rank, doc) in enumerate(zip(rank_mapping, ordered_results), 1):
        # source= attribute uses original rank for citation purposes
        attrs = [f'source="{rank}"', f'id="{rank}"']
        if doc.get("course_id"):
            attrs.append(f'course="{doc["course_id"]}"')
        if doc.get("section"):
            attrs.append(f'section="{doc["section"]}"')
        if doc.get("category"):
            attrs.append(f'category="{doc["category"]}"')
        if doc.get("instructor_name"):
            attrs.append(f'instructor="{doc["instructor_name"]}"')
        if doc.get("term"):
            attrs.append(f'term="{doc["term"]}"')

        attr_str = " ".join(attrs)
        title = doc.get("title", "")
        content = doc.get("content", doc.get("policy_name", ""))

        # XML escape special characters in content
        content_escaped = html.escape(content)
        title_escaped = html.escape(title) if title else ""

        parts.append(f"<chunk {attr_str}>")
        if title_escaped:
            parts.append(f"<title>{title_escaped}</title>")
        parts.append(f"<content>{content_escaped}</content>")
        parts.append("</chunk>")
    parts.append("</context>")
    return "\n".join(parts)


def collapse_whitespace(text: str) -> str:
    """Collapse 3+ consecutive spaces to a single space.

    Gemini sometimes pads markdown table cells with excessive whitespace.
    """
    return re.sub(r' {3,}', ' ', text)
