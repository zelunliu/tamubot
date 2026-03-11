.PHONY: run scrape-catalog scrape-classes scrape-simple-syllabus setup-atlas ingest ingest-dept \
        ingest-corpus test typecheck lint format eval-router probe probe-full \
        eval-draft import-draft bench bench-ragas validate-ragas

# --- App ---
run:
	@echo "Starting TamuBot..."
	@. .venv/bin/activate && streamlit run app.py

# --- Data Pipeline ---
scrape-catalog:
	cd tamu_data/scraper && scrapy crawl catalog

scrape-classes:
	cd tamu_data/scraper && scrapy crawl class_search

scrape-simple-syllabus:
	python tamu_data/scraper/download_simple_syllabus.py

setup-atlas:
	python -m ingestion_pipeline.setup_atlas

ingest:
	python -m ingestion_pipeline.ingest

ingest-dept:
	python -m ingestion_pipeline.ingest --department $(DEPT)

ingest-corpus:
	python -m ingestion_pipeline.ingest --crns-file tamu_data/evals/eval_corpus.json

# --- Dev / Testing ---
test:
	pytest tests/ -v

typecheck:
	mypy rag/ ingestion_pipeline/ evals/ --ignore-missing-imports

lint:
	ruff check rag/ ingestion_pipeline/ evals/ app.py config.py

format:
	ruff format rag/ ingestion_pipeline/ evals/ app.py config.py

eval-router:
	python evals/eval_router_metrics.py

probe:
	python evals/run_probe.py --suite smoke

probe-full:
	python evals/run_probe.py --suite all

# --- Benchmarking ---
eval-draft:
	python evals/generate_eval_draft.py --n 60

import-draft:
	python evals/import_eval_draft.py --draft $(DRAFT) --tag $(or $(TAG),v1)

bench:
	python evals/run_benchmark.py --golden-set $(GOLDEN) --experiment-name $(EXP)

bench-ragas:
	python evals/run_benchmark.py --golden-set $(GOLDEN) --experiment-name $(EXP) --ragas

validate-ragas:
	python evals/validate_ragas.py --benchmark $(BENCH)
