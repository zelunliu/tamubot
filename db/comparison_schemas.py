"""Pydantic schemas for multi-course comparison extraction.

Used in the single-call comparison architecture: Gemini extracts structured
per-course data via CourseComparisonTable; Markdown is rendered in Python
from that data (no second LLM call needed).
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
    course_overview: str = Field(
        default="",
        description="Course description, scope, and general difficulty level from the syllabus",
    )
    learning_outcomes: str = Field(
        default="",
        description="Key learning objectives and competencies students are expected to achieve",
    )
    topics_complexity: str = Field(
        default="",
        description="Key topics covered and their technical depth or difficulty level",
    )
    materials: str = Field(
        default="",
        description="Required textbooks, software, or resources that signal course level",
    )


class CourseComparisonTable(BaseModel):
    """Wrapper for multiple courses being compared.

    Used to structure the Call 1 output when comparing multiple courses.
    """

    courses: list[CourseComparisonData] = Field(
        ...,
        description="List of course comparison data objects, one per course",
    )
