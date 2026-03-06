"""
Download syllabi from tamu.simplesyllabus.com for CSCE + ISEN, all sections,
Fall 2025 and Spring 2026.

Uses Playwright to acquire session cookies (site is JS-gated behind CloudFront),
then switches to requests for efficient pagination + PDF download.

Output:
  tamu_data/raw/simple_syllabus_YYYYMMDD/
    *.pdf
    simple_syllabus_metadata.json
"""

import re
import os
import json
import time
import datetime
import requests
from urllib.parse import quote
from playwright.sync_api import sync_playwright, Page

# ── Config ───────────────────────────────────────────────────────────────────

DEPARTMENTS  = {'CSCE', 'ISEN'}
TARGET_TERMS = {'Fall 2025', 'Spring 2026'}
PAGE_SIZE    = 50
DELAY        = 1.0   # seconds between requests

BASE    = 'https://tamu.simplesyllabus.com'
API     = f'{BASE}/api2/doc-library-search'

# Term entity_ids — filter server-side, avoids scanning all 11k items
_TERM_IDS = {
    'Fall 2025':   'ecd304d6-7795-4f49-a0ee-1d6137884ac7',
    'Spring 2026': '3a9c109e-8e72-4682-966e-b6c754c0596f',
}
_TERM_ID_QS = '&'.join(f'term_ids[]={tid}' for tid in _TERM_IDS.values())
PARAMS  = f'{_TERM_ID_QS}&page_size={{ps}}&page={{pg}}'

TERM_SEMESTER = {'spring': '11', 'summer': '21', 'fall': '41'}

_REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_DATE       = datetime.date.today().strftime('%Y%m%d')
OUTPUT_DIR  = os.path.join(_REPO_ROOT, 'tamu_data', 'raw', f'simple_syllabus_{_DATE}')

# Title pattern: "CSCE 670 600 (46627)"
_TITLE_RE = re.compile(
    r'^(?P<subject>[A-Z]+)\s+(?P<course>\d+)\s+(?P<section>\S+)\s+\((?P<crn>\d+)\)$'
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def term_to_code(term: str) -> str:
    """'Spring 2026' → '202611'"""
    parts = term.lower().split()
    return f'{parts[1]}{TERM_SEMESTER.get(parts[0], "00")}' if len(parts) == 2 else 'unknown'

def term_to_slug_prefix(term_name: str) -> str:
    """'Spring 2026 - College Station' → 'Spring-2026-College-Station'"""
    return term_name.replace(' - ', '-').replace(' ', '-')

def build_slug(term_name: str, subject: str, course: str, section: str, crn: str) -> str:
    prefix = term_to_slug_prefix(term_name)
    return f'{prefix}-{subject}-{course}-{section}-({crn})'

def parse_title(title: str):
    m = _TITLE_RE.match(title.strip())
    return m.groupdict() if m else None

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    meta_path = os.path.join(OUTPUT_DIR, 'simple_syllabus_metadata.json')

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()

        # ── Phase 1: seed session + scan ────────────────────────────────────
        print('Seeding session…')
        page.goto(f'{BASE}/en-US/syllabus-library', wait_until='networkidle')

        print('Scanning syllabi…')
        candidates = []
        pg = 0
        total = None

        while True:
            url = f'{API}?{PARAMS.format(ps=PAGE_SIZE, pg=pg)}'
            data = page.evaluate(f'''async () => {{
                const r = await fetch("{url}", {{headers:{{Accept:"application/json"}}}});
                return await r.json();
            }}''')

            items = data.get('items', [])
            if total is None:
                total = data['pagination']['total']
                pages = -(-total // PAGE_SIZE)
                print(f'  Total: {total} ({pages} pages)')

            for item in items:
                title     = item.get('title', '')
                term_name = item.get('term_name', '')
                code      = item.get('code', '')

                term = term_name.split(' - ')[0] if ' - ' in term_name else term_name
                if term not in TARGET_TERMS:
                    continue
                if 'College Station' not in term_name:
                    continue

                parsed = parse_title(title)
                if not parsed:
                    continue
                if parsed['subject'] not in DEPARTMENTS:
                    continue

                candidates.append({'code': code, 'term_name': term_name, 'term': term, **parsed})

            fetched = pg * PAGE_SIZE + len(items)
            print(f'  Page {pg}: {fetched}/{total} scanned, {len(candidates)} matches')

            if fetched >= total or not items:
                break
            pg += 1
            time.sleep(DELAY)

        print(f'\nFound {len(candidates)} syllabi to print as PDF.')

        # ── Phase 2: print each syllabus view page to PDF ───────────────────
        metadata = {}
        for i, c in enumerate(candidates, 1):
            subject   = c['subject']
            course    = c['course']
            section   = c['section']
            crn       = c['crn']
            code      = c['code']
            term_name = c['term_name']
            term_code = term_to_code(c['term'])

            slug      = build_slug(term_name, subject, course, section, crn)
            slug_enc  = quote(slug, safe='-')
            view_url  = f'{BASE}/en-US/doc/{code}/{slug_enc}?mode=view'
            filename  = f'{term_code}_{subject}_{course}_{section}_{crn}.pdf'
            out_path  = os.path.join(OUTPUT_DIR, filename)

            metadata[filename] = {'syllabus_url': view_url, 'doc_id': code}

            if os.path.exists(out_path):
                print(f'[{i}/{len(candidates)}] Skip (exists): {filename}')
                continue

            print(f'[{i}/{len(candidates)}] Printing {filename}...', end=' ', flush=True)
            try:
                page.goto(view_url, wait_until='networkidle', timeout=30000)
                page.pdf(path=out_path, format='Letter', print_background=True)
                size_kb = os.path.getsize(out_path) // 1024
                print(f'OK ({size_kb}KB)')
            except Exception as e:
                print(f'ERROR: {e}')

            time.sleep(DELAY)

        browser.close()

    # Write metadata
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f'\nDone. Metadata: {meta_path}')
    print(f'Output dir:     {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
