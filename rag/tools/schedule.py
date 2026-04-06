"""Schedule parsing and conflict detection for recursive course recommendations."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MeetingInterval:
    days: frozenset  # e.g. frozenset({'M', 'W'})
    start_min: int   # minutes since midnight
    end_min: int


_VALID_DAY_CHARS = frozenset("MTWRFSU")

_TIME_RE = re.compile(
    r"([MTWRFSU]+)\s+(\d{1,2}:\d{2}\s*(?:AM|PM))\s*[-\u2013]\s*(\d{1,2}:\d{2}\s*(?:AM|PM))",
    re.IGNORECASE,
)


def _parse_days(day_str: str) -> frozenset:
    return frozenset(c for c in day_str.upper() if c in _VALID_DAY_CHARS)


def _to_minutes(time_str: str) -> int:
    t = time_str.upper().replace(" ", "")
    is_pm = t.endswith("PM")
    t = t[:-2]
    h, m = map(int, t.split(":"))
    if is_pm and h != 12:
        h += 12
    elif not is_pm and h == 12:
        h = 0
    return h * 60 + m


def parse_meeting_times(mt_string: str | None) -> MeetingInterval | None:
    """Parse a courses_v3 meeting_times string into a MeetingInterval.

    Returns None for async/online courses or unparseable strings.

    Examples:
        "MW 4:10PM - 5:25PM"  → MeetingInterval(days={'M','W'}, start_min=970, end_min=1045)
        "TR 9:35AM - 10:50AM" → MeetingInterval(days={'T','R'}, start_min=575, end_min=650)
        None / "WEB ASYNC"    → None
    """
    if not mt_string:
        return None
    m = _TIME_RE.search(mt_string)
    if not m:
        return None
    days = _parse_days(m.group(1))
    start = _to_minutes(m.group(2))
    end = _to_minutes(m.group(3))
    if not days or start >= end:
        return None
    return MeetingInterval(days=days, start_min=start, end_min=end)


def schedules_conflict(a: MeetingInterval, b: MeetingInterval) -> bool:
    """True if two intervals share at least one day AND their times overlap."""
    if not (a.days & b.days):
        return False
    return a.start_min < b.end_min and b.start_min < a.end_min


def filter_conflicting_courses(
    chunks: list[dict],
    anchor_interval: MeetingInterval,
    meeting_times_map: dict[str, str | None],
) -> tuple[list[dict], list[str]]:
    """Remove discovery chunks whose course conflicts with anchor_interval.

    Args:
        chunks:            Discovery chunks to filter.
        anchor_interval:   Parsed schedule of the anchor course.
        meeting_times_map: course_id → meeting_times string (from courses_v3).

    Returns:
        (kept_chunks, sorted_conflicted_course_ids)
        Courses with None/unparseable meeting_times are kept (benefit of the doubt).
    """
    conflicted: set[str] = set()
    for cid, mt_str in meeting_times_map.items():
        interval = parse_meeting_times(mt_str)
        if interval and schedules_conflict(anchor_interval, interval):
            conflicted.add(cid)
    kept = [c for c in chunks if c.get("course_id") not in conflicted]
    return kept, sorted(conflicted)
