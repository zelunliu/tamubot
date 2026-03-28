"""Cache utilities for the v4 session cache layer."""
from __future__ import annotations

import re


def normalize_query(query: str) -> str:
    """Normalize a query string for exact-match cache lookups.

    Lowercases, strips punctuation, collapses whitespace.
    """
    cleaned = re.sub(r"[^\w\s]", "", query.lower())
    return re.sub(r"\s+", " ", cleaned).strip()
