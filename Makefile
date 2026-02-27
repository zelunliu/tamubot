.PHONY: run scrape-catalog scrape-classes convert standardize ingest \
        test typecheck lint format eval-router

# --- App ---
run:
	@echo "Starting TamuBot..."
	@. .venv/bin/activate && streamlit run app.py

# --- Data Pipeline ---
scrape-catalog:
	cd tamu_data/scraper && scrapy crawl catalog

scrape-classes:
	cd tamu_data/scraper && scrapy crawl class_search

convert:
	python convert_for_vertex.py

standardize:
	python standardize_syllabi.py

ingest:
	python tamu_data/ingestion/upload_to_corpus.py

# --- Dev / Testing ---
test:
	pytest tests/ -v

typecheck:
	mypy db/ --ignore-missing-imports

lint:
	ruff check db/ app.py config.py

format:
	ruff format db/ app.py config.py

eval-router:
	python evals/eval_router_metrics.py
