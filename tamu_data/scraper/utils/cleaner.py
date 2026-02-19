import trafilatura

def clean_html_content(html_content):
    """
    Extracts main text from HTML content using Trafilatura.
    Removes navigation, footers, and sidebars.
    """
    if not html_content:
        return ""
    
    # trafilatura.extract returns None if extraction fails
    text = trafilatura.extract(html_content, include_comments=False, include_tables=True, no_fallback=True)
    return text if text else ""
