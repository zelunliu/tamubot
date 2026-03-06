import json
import os

from items import SimpleSyllabusItem
from scrapy import Request
from scrapy.pipelines.files import FilesPipeline


class SyllabusPipeline(FilesPipeline):
    def get_media_requests(self, item, info):
        adapter = item
        if adapter.get('file_urls'):
            for file_url in adapter['file_urls']:
                # Pass the item metadata to the request
                yield Request(file_url, meta={'item': item})

    def file_path(self, request, response=None, info=None, *, item=None):
        # Retrieve the item from meta (Scrapy < 2.0 passed item directly, newer uses meta or item arg)
        # We ensure it's passed in get_media_requests via meta
        item = request.meta.get('item')
        
        # Safe filename construction
        term = item.get('term_code', 'unknown')
        subject = item.get('subject', 'UNK')
        course = item.get('course', '000')
        section = item.get('section', '000')
        crn = item.get('crn', '00000')
        
        filename = f"{term}_{subject}_{course}_{section}_{crn}.pdf"
        
        # Return path relative to FILES_STORE
        return filename

class ManifestPipeline:
    """Records syllabus_url + doc_id for each downloaded Simple Syllabus PDF."""

    def open_spider(self, spider):
        self._records = {}
        store = spider.settings.get('FILES_STORE')
        os.makedirs(store, exist_ok=True)
        self._manifest_path = os.path.join(store, 'simple_syllabus_metadata.json')

    def process_item(self, item, spider):
        if not isinstance(item, SimpleSyllabusItem):
            return item

        term_code = item.get('term_code', 'unknown')
        subject   = item.get('subject', 'UNK')
        course    = item.get('course', '000')
        section   = item.get('section', '000')
        crn       = item.get('crn', '')
        filename  = f"{term_code}_{subject}_{course}_{section}_{crn}.pdf"

        self._records[filename] = {
            'syllabus_url': item.get('syllabus_url', ''),
            'doc_id':       item.get('doc_id', ''),
        }
        return item

    def close_spider(self, spider):
        with open(self._manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self._records, f, indent=2)
        spider.logger.info(f'ManifestPipeline: wrote {len(self._records)} entries to {self._manifest_path}')


class ProgressPipeline:
    def __init__(self):
        self.log_path = os.path.join('logs', 'progress_log.txt')
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)

    def process_item(self, item, spider):
        if spider.name == 'catalog' and 'url' in item:
            url = item['url']
            # Only log TamuPageItem (pages), not PdfManifestItem
            if 'content' in item:
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(url + '\n')
        return item