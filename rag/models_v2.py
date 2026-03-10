"""Pydantic v2 models for the V2 pipeline collections: chunks_v2, courses_v2."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from rag.models import Instructor


class ChunkDocV2(BaseModel):
    """One document in the *chunks_v2* collection.

    Structural chunks from header-based segmentation — no category field.
    """
    # Unique key for idempotent upserts: crn + chunk_index
    crn: str
    chunk_index: int

    # Structural header metadata
    header_text: Optional[str] = None
    header_level: int = 0          # 0 = no-header / paragraph-merge fallback
    parent_header: Optional[str] = None

    # Chunk content
    content: str
    has_table: bool = False

    # Contextual anchor prepended before embedding
    anchor: str = ""

    # Denormalized course metadata
    course_id: str
    section: str
    term: str
    instructor_name: Optional[str] = None

    # Embedding (populated during ingestion)
    embedding: Optional[list[float]] = None

    # Housekeeping
    pipeline_version: str = "v2"
    source_file: str = ""
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class CourseDocV2(BaseModel):
    """One document per course section (CRN) in *courses_v2*."""
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
    header_count: int = 0          # chunks with a real header (level > 0)
    syllabus_url: Optional[str] = None
    doc_id: Optional[str] = None
    pipeline_version: str = "v2"
    source_file: str = ""
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
