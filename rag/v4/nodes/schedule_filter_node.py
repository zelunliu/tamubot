"""Schedule filter node — removes discovery courses with conflicting meeting times."""
from __future__ import annotations
from typing import Any
from rag.v4.state import PipelineState


def schedule_filter_node(state: PipelineState, registry: Any) -> dict:
    """Filter conflicting courses from discovery chunks. Only runs for recurrent path."""
    course_ids = state.get("course_ids", [])
    discovery_chunks = state.get("discovery_chunks", [])
    node_trace = list(state.get("node_trace", []))
    node_trace.append("schedule_filter")

    try:
        meeting_times = registry.retriever.get_meeting_times(course_ids)

        from rag.schedule import filter_conflicting_courses, parse_meeting_times

        anchor_interval = None
        for cid in course_ids:
            anchor_interval = parse_meeting_times(meeting_times.get(cid))
            if anchor_interval:
                break

        if anchor_interval is None:
            # Anchor is async — nothing to filter
            return {"discovery_chunks": discovery_chunks, "conflicted_course_ids": [], "node_trace": node_trace}

        disc_cids = list({c.get("course_id") for c in discovery_chunks if c.get("course_id")})
        disc_mt_map = registry.retriever.get_meeting_times(disc_cids)
        filtered, conflicted_ids = filter_conflicting_courses(discovery_chunks, anchor_interval, disc_mt_map)
        return {
            "discovery_chunks": filtered,
            "conflicted_course_ids": conflicted_ids,
            "node_trace": node_trace,
        }
    except Exception as e:
        return {
            "discovery_chunks": discovery_chunks,
            "conflicted_course_ids": [],
            "error": f"Schedule filter failed: {e}",
            "node_trace": node_trace,
        }
