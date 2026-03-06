import scrapy

class SimpleSyllabusItem(scrapy.Item):
    term_code    = scrapy.Field()
    crn          = scrapy.Field()
    subject      = scrapy.Field()
    course       = scrapy.Field()
    section      = scrapy.Field()
    doc_id       = scrapy.Field()
    syllabus_url = scrapy.Field()   # view URL — metadata only
    file_urls    = scrapy.Field()   # PDF URL — for SyllabusPipeline
    files        = scrapy.Field()

class TamuPageItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()

class PdfManifestItem(scrapy.Item):
    url = scrapy.Field()
    program_name = scrapy.Field() # Text of the link or inferred context
    department = scrapy.Field() # Inferred from breadcrumbs or page title
    yield_time = scrapy.Field()

class ClassSectionItem(scrapy.Item):
    term_code = scrapy.Field()
    crn = scrapy.Field()
    title = scrapy.Field()
    subject = scrapy.Field()
    course = scrapy.Field()
    section = scrapy.Field()
    instructor = scrapy.Field()
    raw_data = scrapy.Field()
    
    # Fields for FilesPipeline
    file_urls = scrapy.Field()
    files = scrapy.Field()
