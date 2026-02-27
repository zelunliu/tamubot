import json
import os
import re
from pathlib import Path

# Paths
INPUT_FILE = os.path.join('tamu_data', 'tamu_scraper', 'data', 'scraped_content.jsonl')
OUTPUT_DIR = os.path.join('tamu_data', 'rag_vertex', 'documents')

def sanitize_filename(title):
    # Remove invalid chars and shorten
    name = re.sub(r'[<>:"/\\|?*]', '', title)
    return name[:150].strip()

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    print(f"Reading from {INPUT_FILE}...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                url = data.get('url', '')
                title = data.get('title', 'Untitled')
                content = data.get('content', '')

                if not content.strip():
                    continue

                # Create Markdown content with Frontmatter for Metadata
                md_content = f"""---
url: {url}
title: "{title}"
---

# {title}

{content}
"""
                # Generate unique filename
                safe_title = sanitize_filename(title)
                filename = f"{safe_title}_{count}.md"
                file_path = os.path.join(OUTPUT_DIR, filename)

                with open(file_path, 'w', encoding='utf-8') as out:
                    out.write(md_content)
                
                count += 1
                if count % 500 == 0:
                    print(f"Processed {count} documents...")

            except Exception as e:
                print(f"Skipping line due to error: {e}")

    print(f"Conversion complete! {count} Markdown files created in '{OUTPUT_DIR}'.")
    print("You can now upload these files to a Google Cloud Storage bucket for ingestion.")

if __name__ == "__main__":
    main()
