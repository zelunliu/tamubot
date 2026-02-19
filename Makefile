.PHONY: run scrape-catalog scrape-classes convert standardize ingest

# --- App ---
run:
	@echo "Starting TamuBot..."
	@. .venv/bin/activate && streamlit run app.py

# --- Data Pipeline ---
scrape-catalog:
	cd tamu_data/tamu_scraper && scrapy crawl catalog

scrape-classes:
	cd tamu_data/tamu_scraper && scrapy crawl class_search

convert:
	python convert_for_vertex.py

standardize:
	python standardize_syllabi.py

ingest:
	python tamu_data/ingestion/upload_to_corpus.py
