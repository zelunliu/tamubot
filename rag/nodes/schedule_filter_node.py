"""Schedule filter node — removes discovery courses with conflicting meeting times."""
from __future__ import annotations

from rag.graph.middleware import error_guard_middleware, timing_middleware, tracing_middleware
from rag.state.pipeline_state import PipelineState


@tracing_middleware
@timing_middleware
@error_guard_middleware
def schedule_filter_node(state: PipelineState) -> dict:
    """Filter conflicting courses from discovery chunks. Only runs for recurrent path."""
    from rag.tools.mongo import get_meeting_times
    from rag.tools.schedule import filter_conflicting_courses, parse_meeting_times

    course_ids = state.get("course_ids", [])
    discovery_chunks = state.get("discovery_chunks", [])
    node_trace = list(state.get("node_trace", []))
    node_trace.append("schedule_filter")

    try:
        meeting_times = get_meeting_times(course_ids)

        # Collect all anchor intervals (one per in-person anchor course)
        anchor_intervals = []
        for cid in course_ids:
            interval = parse_meeting_times(meeting_times.get(cid))
            if interval:
                anchor_intervals.append(interval)

        if not anchor_intervals:
            # All anchors are async — nothing to filter
            return {"discovery_chunks": discovery_chunks, "conflicted_course_ids": [], "node_trace": node_trace}

        disc_cids = list({c.get("course_id") for c in discovery_chunks if c.get("course_id")})
        disc_mt_map = get_meeting_times(disc_cids)

        # A discovery course is filtered if it conflicts with ANY anchor interval
        all_conflicted_ids: set[str] = set()
        for anchor_interval in anchor_intervals:
            _, conflicted = filter_conflicting_courses(discovery_chunks, anchor_interval, disc_mt_map)
            all_conflicted_ids.update(conflicted)

        filtered = [c for c in discovery_chunks if c.get("course_id") not in all_conflicted_ids]
        return {
            "discovery_chunks": filtered,
            "conflicted_course_ids": sorted(all_conflicted_ids),
            "node_trace": node_trace,
        }
    except Exception as e:
        return {
            "discovery_chunks": discovery_chunks,
            "conflicted_course_ids": [],
            "error": f"Schedule filter failed: {e}",
            "node_trace": node_trace,
        }
