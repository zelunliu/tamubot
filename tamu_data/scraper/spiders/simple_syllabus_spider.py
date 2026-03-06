import datetime
import json
import os
import re

import scrapy
from items import SimpleSyllabusItem

DEPARTMENTS = ['CSCE', 'ISEN']
TARGET_TERMS = {'Fall 2025', 'Spring 2026'}

_SEMESTER_CODE = {'spring': '11', 'summer': '21', 'fall': '41'}

def _term_to_code(term: str) -> str:
    """'Spring 2026' → '202611', 'Fall 2025' → '202541', etc."""
    parts = term.lower().split()
    if len(parts) == 2:
        semester, year = parts
        code = _SEMESTER_CODE.get(semester, '00')
        return f'{year}{code}'
    return 'unknown'

_SLUG_RE = re.compile(
    r'^(?P<term>.+?)-(?P<location>.+?)-(?P<subject>[A-Z]+)-(?P<course>\d+)-(?P<section>\S+?)(?:-\((?P<crn>\d+)\))?$'
)

BASE = 'https://tamu.simplesyllabus.com'
LIBRARY_URL = f'{BASE}/en-US/syllabus-library'
SEARCH_URL = f'{BASE}/api2/search'

# Absolute path: <repo_root>/tamu_data/raw/simple_syllabus_YYYYMMDD
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_DATE_SUFFIX = datetime.date.today().strftime('%Y%m%d')
OUTPUT_DIR = os.path.join(_REPO_ROOT, 'tamu_data', 'raw', f'simple_syllabus_{_DATE_SUFFIX}')


def _parse_slug(slug: str) -> dict:
    m = _SLUG_RE.match(slug)
    if not m:
        return {}
    return m.groupdict()


class SimpleSyllabusSpider(scrapy.Spider):
    name = 'simple_syllabus'
    allowed_domains = ['tamu.simplesyllabus.com']

    custom_settings = {
        'CONCURRENT_REQUESTS': 1,
        'DOWNLOAD_DELAY': 1.5,
        'ROBOTSTXT_OBEY': False,
        'FILES_STORE': OUTPUT_DIR,
        'HTTPERROR_ALLOWED_CODES': [500],  # let 500s reach our callback for diagnosis
    }

    def start_requests(self):
        # Seed session cookies by visiting the library page first
        yield scrapy.Request(
            url=LIBRARY_URL,
            callback=self.after_seed,
            dont_filter=True,
        )

    def after_seed(self, response):
        self.logger.info(f'Library page status={response.status}, cookies={response.headers.getlist("Set-Cookie")}')
        for dept in DEPARTMENTS:
            yield self._search_request(dept, offset=0)

    def _search_request(self, dept: str, offset: int):
        url = (
            f'{SEARCH_URL}?q={dept}&locale=en-US&limit=100&offset={offset}'
        )
        return scrapy.Request(
            url=url,
            headers={
                'Accept': 'application/json',
                'Referer': LIBRARY_URL,
            },
            callback=self.parse_search,
            meta={'dept': dept, 'offset': offset},
            dont_filter=True,
        )

    def parse_search(self, response):
        dept = response.meta['dept']
        offset = response.meta['offset']

        if response.status == 500:
            self.logger.error(
                f'Search API 500 for dept={dept} offset={offset}. '
                f'Response body: {response.text[:500]}'
            )
            return

        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            self.logger.error(f'Non-JSON response for dept={dept} offset={offset}: {response.text[:200]}')
            return

        results = data.get('results', [])
        total = data.get('total', 0)

        self.logger.info(f'dept={dept} offset={offset} total={total} got={len(results)}')

        for item_data in results:
            term = item_data.get('term', '')
            location = item_data.get('location', '')

            if term not in TARGET_TERMS:
                continue
            if 'College Station' not in location:
                continue

            doc_id = item_data.get('id') or item_data.get('docId') or item_data.get('doc_id')
            slug = item_data.get('slug', '')

            if not doc_id or not slug:
                self.logger.warning(f'Missing doc_id or slug in result: {item_data}')
                continue

            parsed = _parse_slug(slug)
            if not parsed:
                self.logger.warning(f'Could not parse slug: {slug}')
                continue

            section = parsed.get('section', '')
            if section != '600':
                continue

            term_code = _term_to_code(term)
            subject = parsed.get('subject') or dept
            course = parsed.get('course', '000')
            crn = parsed.get('crn', '')

            pdf_url = f'{BASE}/api2/doc-pdf/{doc_id}/{slug}.pdf'
            view_url = f'{BASE}/en-US/doc/{doc_id}/{slug}?mode=view'

            scrapy_item = SimpleSyllabusItem()
            scrapy_item['term_code'] = term_code
            scrapy_item['crn'] = crn
            scrapy_item['subject'] = subject
            scrapy_item['course'] = course
            scrapy_item['section'] = section
            scrapy_item['doc_id'] = doc_id
            scrapy_item['syllabus_url'] = view_url
            scrapy_item['file_urls'] = [pdf_url]

            yield scrapy_item

        # Paginate
        next_offset = offset + 100
        if next_offset < total:
            yield self._search_request(dept, next_offset)
