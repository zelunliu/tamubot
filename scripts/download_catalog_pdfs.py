import json
import os
import requests
import time
from urllib.parse import urlparse

MANIFEST_FILE = 'tamu_data/tamu_scraper/data/pdf_manifest.json'
OUTPUT_DIR = 'tamu_data/tamu_scraper/catalog_pdfs'

def download_pdfs():
    if not os.path.exists(MANIFEST_FILE):
        print("Manifest not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load manifest and deduplicate URLs
    urls = {}
    with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            url = data['url']
            if url not in urls:
                urls[url] = data

    print(f"Found {len(urls)} unique PDFs to download.")
    
    headers = {'User-Agent': 'TAMU-Student-Project-Research'}
    
    count = 0
    for url, meta in urls.items():
        # Create a safe filename
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename.endswith('.pdf'):
            filename += '.pdf'
            
        # If common names like 'homepage.pdf', prefix with a hash or part of path
        path_slug = parsed.path.replace('/', '_').strip('_')
        full_filename = f"{path_slug}"
        if not full_filename.endswith('.pdf'):
            full_filename += '.pdf'
            
        file_path = os.path.join(OUTPUT_DIR, full_filename)
        
        if os.path.exists(file_path):
            count += 1
            continue

        try:
            print(f"Downloading: {url}")
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(resp.content)
                count += 1
                if count % 10 == 0:
                    print(f"Progress: {count}/{len(urls)}")
                # Politeness delay
                time.sleep(1)
            else:
                print(f"Failed {url}: Status {resp.status_code}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

    print(f"Finished! {count} PDFs are now in {OUTPUT_DIR}")

if __name__ == "__main__":
    download_pdfs()
