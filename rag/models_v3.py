"""Pydantic v2 models for the V3 pipeline collections: chunks_v3, courses_v3."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from rag.models import Instructor


class ChunkDocV3(BaseModel):
    """One document in the *chunks_v3* collection.

    Flat token chunks (~600 tokens each) — no header or category fields.
    """
    # Unique key for idempotent upserts: crn + chunk_index
    crn: str
    chunk_index: int

    # Chunk content
    content: str
    has_table: bool = False

    # Denormalized course metadata
    course_id: str
    section: str
    term: str
    instructor_name: Optional[str] = None

    # Embedding (populated during ingestion)
    embedding: Optional[list[float]] = None

    # Housekeeping
    pipeline_version: str = "v3"
    source_file: str = ""
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class CourseDocV3(BaseModel):
    """One document per course section (CRN) in *courses_v3*."""
    crn: str
    course_id: str
    section: str
    term: str
    instructor: Optional[Instructor] = None
    teaching_assistants: list[str] = Field(default_factory=list)
    meeting_times: Optional[str] = None
    location: Optional[str] = None
    credit_hours: Optional[str] = None
    chunk_count: int = 0
    syllabus_url: Optional[str] = None
    doc_id: Optional[str] = None
    pipeline_version: str = "v3"
    source_file: str = ""
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
