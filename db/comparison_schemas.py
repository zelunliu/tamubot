"""Pydantic schemas for multi-course comparison extraction.

Used in the two-call architecture for structured JSON output during
course comparisons (Call 1: extract per-course data, Call 2: synthesize to Markdown).
"""

from pydantic import BaseModel, Field


class CourseComparisonData(BaseModel):
    """Structured data extracted from a single course for comparison purposes.

    Used as the response schema for Call 1 of the two-call comparison architecture.
    Enforces structured JSON output from Gemini for deterministic parsing.
    """

    course_id: str = Field(
        ...,
        description="Course identifier (e.g., CSCE-638)",
    )
    grading: str = Field(
        ...,
        description="Grading breakdown or weighting (e.g., 'Exams 40%, Projects 30%, Participation 30%')",
    )
    workload: str = Field(
        ...,
        description="Workload or course difficulty assessment (e.g., 'Heavy - expect 15-20 hours/week')",
    )
    prerequisites: str = Field(
        ...,
        description="Prerequisites or recommended background (e.g., 'CSCE 121, CSCE 222')",
    )
    section: str = Field(
        default="",
        description="Course section identifier (e.g., '001', 'SLI'). Optional.",
    )
    instructor: str = Field(
        default="",
        description="Instructor name. Optional.",
    )


class CourseComparisonTable(BaseModel):
    """Wrapper for multiple courses being compared.

    Used to structure the Call 1 output when comparing multiple courses.
    """

    courses: list[CourseComparisonData] = Field(
        ...,
        description="List of course comparison data objects, one per course",
    )
