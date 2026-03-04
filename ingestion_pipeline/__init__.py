"""Ingestion pipeline — PDF parsing, embedding, Atlas setup.

Public API (import from here, not submodules):
    from ingestion_pipeline import parse_pdf, run_ingest, setup_indexes
"""

__all__ = ["parse_pdf", "run_ingest", "setup_indexes"]


def __getattr__(name: str):
    if name == "parse_pdf":
        from ingestion_pipeline.process_syllabi import parse_pdf
        return parse_pdf
    if name == "run_ingest":
        from ingestion_pipeline.ingest import main
        return main
    if name == "setup_indexes":
        from ingestion_pipeline.setup_atlas import main
        return main
    raise AttributeError(f"module 'ingestion_pipeline' has no attribute {name!r}")
