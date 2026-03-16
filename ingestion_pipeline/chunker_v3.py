"""
ingestion_pipeline/chunker_v3.py

Flat token-based chunker for the V3 syllabus pipeline.

No header parsing — takes plain text (Markdown markers stripped) and produces
size-equalized overlapping chunks via a two-pass algorithm:

  Pass 1: count total tokens T, compute K = ceil(T / chunk_size)
  Pass 2: greedily assign paragraphs into K equal-token groups
          (target ≈ T/K unique tokens per group)
  Output: chunk i = overlap_tail_from_group[i-1] + group[i]

This guarantees exactly K chunks where each group contributes ≈ T/K unique
tokens — so the last chunk is never shorter than the others.

Chunk schema:
    {
        "chunk_index": int,
        "content": str,
        "has_table": bool,
    }
"""

import math
import re

# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokens_approx(text: str) -> int:
    return max(0, round(len(text) / 4))


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; fall back to single newlines when no blanks exist."""
    parts = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in text.splitlines() if p.strip()]
    return parts


def _has_table(content: str) -> bool:
    return bool(re.search(r"\|.*\|", content))


def _strip_markdown(text: str) -> str:
    """Remove Markdown heading markers (#, ##, etc.)."""
    lines = []
    for line in text.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            lines.append(m.group(1))
        else:
            lines.append(line)
    return "\n".join(lines)


def _build_overlap(parts: list[str], overlap: int) -> list[str]:
    """Return tail of parts whose combined tokens reach ~overlap."""
    result: list[str] = []
    total = 0
    for p in reversed(parts):
        t = _tokens_approx(p)
        if total + t > overlap and result:
            break
        result.insert(0, p)
        total += t
    return result


# ── Core chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[dict]:
    """Two-pass equalized chunker.

    Pass 1: split text into paragraphs, count tokens, decide K = ceil(T/chunk_size).
    Pass 2: greedily fill K groups with ≈ T/K unique tokens each.
    Output: chunk i = overlap_tail(group[i-1]) + group[i].

    All chunks have approximately equal size. The first chunk has no overlap
    prefix (nothing before it); all others gain ~overlap tokens of context.

    Returns a list of chunk dicts with keys: chunk_index, content, has_table.
    """
    clean = _strip_markdown(text)
    paragraphs = _split_paragraphs(clean)
    if not paragraphs:
        return []

    total_tokens = sum(_tokens_approx(p) for p in paragraphs)
    if total_tokens == 0:
        return []

    k = max(1, math.ceil(total_tokens / chunk_size))
    target_unique = total_tokens / k

    # ── Pass 2: partition paragraphs into k groups ────────────────────────────
    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0.0

    for para in paragraphs:
        current.append(para)
        current_tokens += _tokens_approx(para)
        # Flush into a new group once we reach the target, unless this is the last group
        if len(groups) < k - 1 and current_tokens >= target_unique:
            groups.append(current)
            current = []
            current_tokens = 0.0

    if current:  # last group (may be empty if perfectly divisible)
        groups.append(current)

    # ── Build chunks: prepend overlap from previous group ─────────────────────
    chunks: list[dict] = []
    for i, group in enumerate(groups):
        if not group:
            continue
        if i > 0 and groups[i - 1]:
            overlap_parts = _build_overlap(groups[i - 1], overlap)
            parts = overlap_parts + group
        else:
            parts = group
        content = "\n\n".join(parts)
        chunks.append({
            "chunk_index": len(chunks),
            "content": content,
            "has_table": _has_table(content),
        })

    return chunks
