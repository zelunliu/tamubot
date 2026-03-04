.PHONY: run scrape-catalog scrape-classes setup-atlas ingest ingest-dept \
        test typecheck lint format eval-router probe probe-full

# --- App ---
run:
	@echo "Starting TamuBot..."
	@. .venv/bin/activate && streamlit run app.py

# --- Data Pipeline ---
scrape-catalog:
	cd tamu_data/scraper && scrapy crawl catalog

scrape-classes:
	cd tamu_data/scraper && scrapy crawl class_search

setup-atlas:
	python -m ingestion_pipeline.setup_atlas

ingest:
	python -m ingestion_pipeline.ingest

ingest-dept:
	python -m ingestion_pipeline.ingest --department $(DEPT)

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
