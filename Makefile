.PHONY: run scrape-catalog scrape-classes scrape-simple-syllabus setup-atlas ingest ingest-dept \
        ingest-corpus test typecheck lint format probe probe-v3 probe-full \
        eval-draft import-draft bench bench-ragas test-v4 probe-v4 \
        eval-chunking sandbox-up sandbox-down sandbox-shell agent

# --- App ---
run:
	@echo "Starting TamuBot..."
	@streamlit run app.py --server.headless true

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

ingest-v3:
	python -m ingestion_pipeline.ingest --v3

ingest-dept:
	python -m ingestion_pipeline.ingest --department $(DEPT)

ingest-corpus:
	python -m ingestion_pipeline.ingest --v3 --crns-file tamu_data/evals/eval_corpus.json

# --- Dev / Testing ---
test:
	pytest tests/ -v

typecheck:
	mypy rag/ ingestion_pipeline/ evals/ --ignore-missing-imports

lint:
	ruff check rag/ ingestion_pipeline/ evals/ app.py config.py

format:
	ruff format rag/ ingestion_pipeline/ evals/ app.py config.py

probe:
	python evals/run_probe.py --suite smoke

probe-v3:
	USE_V4_PIPELINE=false python evals/run_probe.py --suite smoke

probe-full:
	python evals/run_probe.py --suite all

test-v4:
	pytest tests/test_v4_*.py -v

probe-v4:
	python evals/run_probe.py --suite smoke

# --- Benchmarking ---
eval-draft:
	python evals/generate_eval_draft.py --n 60

import-draft:
	python evals/import_eval_draft.py --draft $(DRAFT) --tag $(or $(TAG),v1)

bench:
	python evals/run_benchmark.py --golden-set $(GOLDEN) --experiment-name $(EXP)

bench-ragas:
	python evals/run_benchmark.py --golden-set $(GOLDEN) --experiment-name $(EXP) --ragas

eval-chunking:
	SESSION_CACHE_ENABLED=false python evals/eval_chunking.py \
		--golden-set $(GOLDEN) \
		--experiment $(EXP) \
		$(if $(RAGAS),--ragas,) \
		$(if $(TOP_K),--top-k $(TOP_K),) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD),) \
		$(if $(CHUNK_SIZE),--chunk-size $(CHUNK_SIZE),) \
		$(if $(CHUNK_OVERLAP),--chunk-overlap $(CHUNK_OVERLAP),) \
		$(if $(DESC),--description "$(DESC)",) \
		$(if $(OUTPUT),--output $(OUTPUT),)

# --- Docker Sandbox ---
sandbox-up:
	docker compose up -d

sandbox-down:
	docker compose down

sandbox-shell:
	docker exec -it tamubot-dev-1 bash

agent:
	docker exec -it tamubot-dev-1 claude --dangerously-skip-permissions
