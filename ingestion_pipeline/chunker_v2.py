"""
ingestion_pipeline/chunker_v2.py

Structural header-based chunking for the V2 syllabus pipeline.

Takes filtered Markdown text (output of strip_markdown_boilerplate) and
produces a list of chunk dicts with header metadata — no LLM classification.

Chunk schema:
    {
        "header_text": str | None,
        "header_level": int,       # 0 = no-header fallback
        "parent_header": str | None,
        "content": str,
        "has_table": bool,
    }
"""

import re

CHUNK_MAX_WORDS = 1000          # split chunks larger than this at para breaks
CHUNK_MIN_WORDS_FOR_MERGE = 200 # merge sibling chunks below this when >30 sections
CHUNK_TARGET_TOKENS = 600       # target size for paragraph-merge fallback
MAX_SECTIONS_BEFORE_MERGE = 30  # trigger sibling-merge when section count exceeds this


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_words(text: str) -> int:
    return len(text.split())


def _tokens_approx(text: str) -> int:
    return max(0, round(len(text) / 4))


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]


def _has_table(content: str) -> bool:
    return bool(re.search(r"\|.*\|", content))


def _parse_md_sections(md_text: str) -> list[tuple[str | None, int, list[str]]]:
    """Parse Markdown into (header, level, content_lines) triples."""
    sections: list[tuple[str | None, int, list[str]]] = []
    cur_header: str | None = None
    cur_level: int = 0
    cur_lines: list[str] = []

    for line in md_text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            sections.append((cur_header, cur_level, cur_lines))
            cur_header = m.group(2).strip()
            cur_level = len(m.group(1))
            cur_lines = []
        else:
            cur_lines.append(line)
    sections.append((cur_header, cur_level, cur_lines))
    return sections


# ── Core chunking ─────────────────────────────────────────────────────────────

def chunk_markdown(md_text: str) -> list[dict]:
    """Convert filtered Markdown into a list of structured chunk dicts.

    Fallbacks:
    - No headers → paragraph merge to ~600 tokens (header_level=0)
    - >30 sections → merge small siblings under 200 words
    - Chunk >1000 words → split at paragraph break
    """
    sections = _parse_md_sections(md_text)
    real_sections = [(h, lv, c) for h, lv, c in sections if h is not None]

    if not real_sections:
        return _paragraph_merge_fallback(md_text)

    chunks = _build_chunks(sections)

    if len(chunks) > MAX_SECTIONS_BEFORE_MERGE:
        chunks = _merge_small_siblings(chunks)

    chunks = _split_large_chunks(chunks)
    return chunks


def _build_chunks(sections: list[tuple]) -> list[dict]:
    """Build chunk dicts with parent tracking from parsed sections."""
    chunks: list[dict] = []
    parent_stack: dict[int, str] = {}  # level → header_text

    for header, level, content_lines in sections:
        content = "\n".join(content_lines).strip()

        if header is None:
            if content:
                chunks.append({
                    "header_text": None,
                    "header_level": 0,
                    "parent_header": None,
                    "content": content,
                    "has_table": _has_table(content),
                })
            continue

        # Update parent stack at this level; invalidate deeper levels
        parent_stack[level] = header
        for l in list(parent_stack):
            if l > level:
                del parent_stack[l]

        # Find nearest ancestor
        parent: str | None = None
        for l in range(level - 1, 0, -1):
            if l in parent_stack:
                parent = parent_stack[l]
                break

        chunks.append({
            "header_text": header,
            "header_level": level,
            "parent_header": parent,
            "content": content,
            "has_table": _has_table(content),
        })

    return [c for c in chunks if c["content"] or c["header_text"]]


def _paragraph_merge_fallback(md_text: str) -> list[dict]:
    """No-header fallback: merge paragraphs into ~600-token chunks."""
    paragraphs = _split_paragraphs(md_text)
    chunks: list[dict] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        t = _tokens_approx(para)
        if current_tokens + t > CHUNK_TARGET_TOKENS and current_parts:
            chunks.append(_make_fallback_chunk("\n\n".join(current_parts)))
            current_parts = [para]
            current_tokens = t
        else:
            current_parts.append(para)
            current_tokens += t

    if current_parts:
        chunks.append(_make_fallback_chunk("\n\n".join(current_parts)))
    return chunks


def _make_fallback_chunk(content: str) -> dict:
    return {
        "header_text": None,
        "header_level": 0,
        "parent_header": None,
        "content": content,
        "has_table": _has_table(content),
    }


def _merge_small_siblings(chunks: list[dict]) -> list[dict]:
    """Merge sibling chunks under CHUNK_MIN_WORDS_FOR_MERGE to reduce count."""
    if not chunks:
        return chunks

    merged = [chunks[0].copy()]
    for chunk in chunks[1:]:
        prev = merged[-1]
        same_parent = prev["parent_header"] == chunk["parent_header"]
        same_level = prev["header_level"] == chunk["header_level"]
        prev_small = _count_words(prev["content"]) < CHUNK_MIN_WORDS_FOR_MERGE

        if same_parent and same_level and prev_small:
            h = chunk["header_text"]
            lv = chunk["header_level"]
            sep = f"\n\n{'#' * lv} {h}\n" if h else "\n\n"
            prev["content"] = (prev["content"] + sep + chunk["content"]).strip()
            prev["has_table"] = prev["has_table"] or chunk["has_table"]
        else:
            merged.append(chunk.copy())

    return merged


def _split_large_chunks(chunks: list[dict]) -> list[dict]:
    """Split chunks over CHUNK_MAX_WORDS at paragraph boundaries."""
    result: list[dict] = []
    for chunk in chunks:
        if _count_words(chunk["content"]) <= CHUNK_MAX_WORDS:
            result.append(chunk)
            continue

        paragraphs = _split_paragraphs(chunk["content"])
        parts: list[str] = []
        current_paras: list[str] = []
        current_words = 0

        for para in paragraphs:
            w = _count_words(para)
            if current_words + w > CHUNK_MAX_WORDS and current_paras:
                parts.append("\n\n".join(current_paras))
                current_paras = [para]
                current_words = w
            else:
                current_paras.append(para)
                current_words += w
        if current_paras:
            parts.append("\n\n".join(current_paras))

        for idx, part in enumerate(parts):
            sub = chunk.copy()
            sub["content"] = part
            sub["has_table"] = _has_table(part)
            if idx > 0 and chunk["header_text"]:
                sub["header_text"] = f"{chunk['header_text']} (cont.)"
            result.append(sub)

    return result
