import os
from urllib.parse import urlparse

import scrapy
from items import PdfManifestItem, TamuPageItem
from utils.cleaner import clean_html_content


class CatalogSpider(scrapy.Spider):
    name = 'catalog'
    allowed_domains = ['catalog.tamu.edu']
    start_urls = ['https://catalog.tamu.edu/']

    def __init__(self, *args, **kwargs):
        super(CatalogSpider, self).__init__(*args, **kwargs)
        self.visited_urls = set()
        self.log_path = os.path.join('logs', 'progress_log.txt')
        
        # Load progress log
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.visited_urls.add(line.strip())
        
        self.logger.info(f"Loaded {len(self.visited_urls)} visited URLs from history.")

    def start_requests(self):
        for url in self.start_urls:
            # We always start from the root to find new/missed links
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        # 1. Process current page content if not visited
        if response.url not in self.visited_urls:
            cleaned_text = clean_html_content(response.body)
            
            page_item = TamuPageItem()
            page_item['url'] = response.url
            page_item['title'] = response.css('title::text').get(default='').strip()
            page_item['content'] = cleaned_text
            yield page_item
            
            # Mark as visited in memory to avoid duplicate items in same session
            self.visited_urls.add(response.url)

        # 2. Extract links (always do this to find new paths)
        for link in response.css('a'):
            href = link.attrib.get('href')
            if not href:
                continue
            
            absolute_url = response.urljoin(href)
            parsed_url = urlparse(absolute_url)

            # Clean fragment from URL for deduplication
            clean_url = absolute_url.split('#')[0]

            # Check for PDF
            if parsed_url.path.lower().endswith('.pdf'):
                pdf_item = PdfManifestItem()
                pdf_item['url'] = absolute_url
                pdf_item['program_name'] = link.css('::text').get(default='').strip()
                # Use current page title as a proxy for department/context
                pdf_item['department'] = response.css('title::text').get(default='').strip()
                yield pdf_item
                continue

            # Check if internal link and not visited
            if parsed_url.netloc == 'catalog.tamu.edu':
                if clean_url not in self.visited_urls:
                    # Note: Scrapy's internal dupe filter handles the current session.
                    # We check our persistent history here.
                    yield scrapy.Request(clean_url, callback=self.parse)
