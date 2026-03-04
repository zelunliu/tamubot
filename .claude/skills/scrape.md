# /scrape — Add or run a web scraper for TAMU data

## Quick reference

| Target | Command |
|---|---|
| Simple Syllabus PDFs | `make scrape-simple-syllabus` |
| HowdyPortal class sections | `make scrape-classes` |
| TAMU catalog pages | `make scrape-catalog` |

---

## When asked to scrape a new site

### Step 1 — Probe the site first

Before writing any code, open a quick Playwright probe:

```python
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page()
    captured = []
    page.on('response', lambda r: captured.append(f'{r.status} {r.url}') if 'target-domain' in r.url else None)
    page.goto('https://target-site/landing-page', wait_until='networkidle')
    for c in captured: print(c)
    b.close()
```

Look for:
- JSON API endpoints (`.../api/...`, `.../api2/...`)
- Auth patterns (`session`, `XSRF-TOKEN` cookies set by JS)
- Pagination fields (`total`, `offset`, `page`, `page_size`)

### Step 2 — Check if requests works or if Playwright is required

```python
import requests
r = requests.get('https://target-site/api/endpoint', headers={'Accept': 'application/json'})
print(r.status_code, r.headers.get('X-Cache'), r.text[:200])
```

- `X-Cache: Error from cloudfront` or HTTP 500 with empty body → **CloudFront WAF blocking** → must use Playwright for all requests
- Cookies set only after JS runs → must use `page.evaluate()` for API calls or extract cookies from `ctx.cookies()` after page load

### Step 3 — Choose the right approach

**Scrapy** (use when):
- Site is not JS-gated (plain HTML or open API)
- Need depth-first crawling, deduplication, robots.txt respect
- Spider goes in `tamu_data/scraper/spiders/`, item in `items.py`

**Standalone Playwright script** (use when):
- Site is JS-gated (CloudFront WAF, session cookies set by JS)
- Need in-browser `page.evaluate()` for API calls
- Need `page.pdf()` to save rendered pages
- Script goes in `tamu_data/scraper/download_<name>.py`
- Add a `make scrape-<name>` target

### Step 4 — Output conventions

- PDFs → `tamu_data/raw/<name>_YYYYMMDD/*.pdf`
- Metadata sidecar → `tamu_data/raw/<name>_YYYYMMDD/<name>_metadata.json`
  ```json
  { "202611_CSCE_670_600_46627.pdf": { "syllabus_url": "...", "doc_id": "..." } }
  ```
- Filename pattern: `{term_code}_{SUBJECT}_{COURSE}_{SECTION}_{CRN}.pdf`
- Term codes: Spring→11, Summer→21, Fall→41  (e.g. Spring 2026 → `202611`)

---

## Lessons learned

### tamu.simplesyllabus.com
- **CloudFront WAF blocks all non-browser requests** — Scrapy + plain requests both return 500
- Real endpoint: `/api2/doc-library-search?term_statuses[]=future&term_statuses[]=current&page_size=50&page=N`
- `/api2/search` and `/api2/doc-pdf` are broken server-side (500 for all docs)
- **Solution**: use Playwright in-page `page.evaluate(fetch(...))` for API calls, `page.pdf()` for saving
- Slug URL-encode parentheses: `(46627)` → `%2846627%29` via `urllib.parse.quote(slug, safe='-')`
- Scraper: `tamu_data/scraper/download_simple_syllabus.py`

### howdyportal.tamu.edu
- Open API, Scrapy works fine
- Seed session: GET the public class search page first, then POST to `/api/course-sections`
- Spider: `tamu_data/scraper/spiders/class_search_spider.py`

### Scrapy project config (flat layout)
- `scrapy.cfg` must say `default = settings` (not `tamu_scraper.settings`)
- `settings.py`: `SPIDER_MODULES = ['spiders']`, pipelines as `'pipelines.ClassName'`
- Spider imports: `from items import X` (not `from ..items import X`)

---

## Examples

```
/scrape                          # ask what to scrape
/scrape simple-syllabus          # re-run Simple Syllabus downloader
/scrape add <url>                # probe a new site and plan a scraper
```
