---
name: scrape
description: Use when scraping a TAMU site, downloading syllabi, or building a new web crawler for TamuBot data
triggers: ["scrape", "download syllabi", "add scraper", "crawl", "scrape classes", "scrape catalog"]
---

# /scrape — TAMU Data Scrapers

## Quick reference

| Target | Command |
|---|---|
| Simple Syllabus PDFs | `make scrape-simple-syllabus` |
| HowdyPortal class sections | `make scrape-classes` |
| TAMU catalog pages | `make scrape-catalog` |

---

## Adding a new scraper

### 1 — Probe the site

```python
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page()
    captured = []
    page.on('response', lambda r: captured.append(f'{r.status} {r.url}') if 'target-domain' in r.url else None)
    page.goto('https://target-site/', wait_until='networkidle')
    for c in captured: print(c)
    b.close()
```

Look for: JSON API endpoints, auth cookies set by JS, pagination fields.

### 2 — Choose approach

**Scrapy** — plain HTML or open API, depth-first crawling. Spider → `tamu_data/scraper/spiders/`.

**Standalone Playwright** — JS-gated sites (CloudFront WAF, session cookies from JS), `page.pdf()` needed. Script → `tamu_data/scraper/download_<name>.py` + `make scrape-<name>` target.

### 3 — Output conventions

- PDFs → `tamu_data/raw/<name>_YYYYMMDD/*.pdf`
- Metadata → `tamu_data/raw/<name>_YYYYMMDD/<name>_metadata.json`
- Filename: `{term_code}_{SUBJECT}_{COURSE}_{SECTION}_{CRN}.pdf`
- Term codes: Spring→11, Summer→21, Fall→41

---

## simplesyllabus.com notes

- CloudFront WAF blocks plain requests → must use Playwright
- Real search endpoint: `/api2/doc-library-search` (NOT `/api2/search`)
- `/api2/doc-pdf` is broken → use `page.pdf()` instead
- Filter by term: `term_ids[]={entity_id}` (NOT `term_statuses[]`)
  - Fall 2025 CS: `ecd304d6-7795-4f49-a0ee-1d6137884ac7`
  - Spring 2026 CS: `3a9c109e-8e72-4682-966e-b6c754c0596f`
- Slug: `{term_name.replace(' - ','-').replace(' ','-')}-{SUBJ}-{COURSE}-{SEC}-({CRN})`; URL-encode parens
- Scraper: `tamu_data/scraper/download_simple_syllabus.py`

## howdyportal.tamu.edu notes

- Open API, Scrapy works. Seed session via GET first, then POST `/api/course-sections`.
- Spider: `tamu_data/scraper/spiders/class_search_spider.py`

## Scrapy flat layout

- `scrapy.cfg`: `default = settings`
- `settings.py`: `SPIDER_MODULES = ['spiders']`
- Spider imports: `from items import X`
