BOT_NAME = 'tamu_scraper'

SPIDER_MODULES = ['spiders']
NEWSPIDER_MODULE = 'spiders'

# User-Agent as requested
USER_AGENT = 'TAMU-Student-Project-Research'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# 1.5 second download delay
DOWNLOAD_DELAY = 1.5

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
   'pipelines.SyllabusPipeline': 1,
   'pipelines.ManifestPipeline': 2,
   'pipelines.ProgressPipeline': 300,
}

FILES_STORE = 'syllabi'


# Request Fingerprinter implementation (standard in newer Scrapy)
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"
