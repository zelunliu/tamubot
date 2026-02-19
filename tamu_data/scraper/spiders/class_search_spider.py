import scrapy
import json
from ..items import ClassSectionItem

class ClassSearchSpider(scrapy.Spider):
    name = 'class_search'
    allowed_domains = ['howdyportal.tamu.edu']
    start_urls = ['https://howdyportal.tamu.edu/uPortal/p/public-class-search-ui.ctf1/max/render.uP']

    custom_settings = {
        'CONCURRENT_REQUESTS': 1,  # Keep it polite
        'DOWNLOAD_DELAY': 2,
    }

    def parse(self, response):
        # We just need the cookies from this initial request
        # Now fetch all terms
        yield scrapy.Request(
            url='https://howdyportal.tamu.edu/api/all-terms',
            callback=self.parse_terms,
            dont_filter=True
        )

    def parse_terms(self, response):
        terms = json.loads(response.text)
        # Sort terms by code descending to get latest first
        terms.sort(key=lambda x: x.get('STVTERM_CODE', ''), reverse=True)
        
        # Filter for Spring 2026 (Code: 202611)
        target_term = '202611'
        
        for term in terms:
            term_code = term.get('STVTERM_CODE')
            if term_code == target_term:
                self.logger.info(f"Queueing target term: {term_code} ({term.get('STVTERM_DESC')})")
                yield scrapy.Request(
                    url='https://howdyportal.tamu.edu/api/course-sections',
                    method='POST',
                    body=json.dumps({"termCode": term_code}),
                    headers={'Content-Type': 'application/json'},
                    callback=self.parse_sections,
                    meta={'term_code': term_code},
                    dont_filter=True
                )

    def parse_sections(self, response):
        term_code = response.meta['term_code']
        sections = json.loads(response.text)
        self.logger.info(f"Processing {len(sections)} sections for term {term_code}")
        
        for sec in sections:
            # Filter for College Station campus
            # The field is usually SWV_CLASS_SEARCH_SITE or SWV_CLASS_SEARCH_ATTRIBUTES
            campus = sec.get('SWV_CLASS_SEARCH_SITE', '')
            if campus != 'College Station':
                continue

            item = ClassSectionItem()
            item['term_code'] = term_code
            item['crn'] = sec.get('SWV_CLASS_SEARCH_CRN')
            item['title'] = sec.get('SWV_CLASS_SEARCH_TITLE')
            item['subject'] = sec.get('SWV_CLASS_SEARCH_SUBJECT')
            item['course'] = sec.get('SWV_CLASS_SEARCH_COURSE')
            item['section'] = sec.get('SWV_CLASS_SEARCH_SECTION')
            item['instructor'] = sec.get('SWV_CLASS_SEARCH_INSTRCTR_JSON')
            # Store the full record for future extraction of secondary fields
            item['raw_data'] = sec
            
            # Check for syllabus
            if sec.get('SWV_CLASS_SEARCH_HAS_SYL_IND') == 'Y':
                # URL format: https://howdyportal.tamu.edu/api/course-syllabus-pdf?termCode=202411&crn=50142
                syllabus_url = f"https://howdyportal.tamu.edu/api/course-syllabus-pdf?termCode={term_code}&crn={item['crn']}"
                item['file_urls'] = [syllabus_url]
            
            yield item
