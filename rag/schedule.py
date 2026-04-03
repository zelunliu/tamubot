"""Backward-compat shim — import from rag.tools.schedule instead."""
from rag.tools.schedule import *  # noqa: F401, F403
from rag.tools.schedule import (
    MeetingInterval, parse_meeting_times, schedules_conflict, filter_conflicting_courses
)
