import os
from pdfminer.high_level import extract_text
import concurrent.futures

SOURCE_DIRS = ['tamu_data/tamu_scraper/syllabi', 'tamu_data/tamu_scraper/catalog_pdfs']
OUTPUT_DIR = 'tamu_data/rag_vertex/extracted_text'

def process_file(file_info):
    src_path, dest_path = file_info
    try:
        text = extract_text(src_path)
        if text.strip():
            with open(dest_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
    return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tasks = []
    for sdir in SOURCE_DIRS:
        if not os.path.exists(sdir):
            continue
        for fname in os.listdir(sdir):
            if fname.lower().endswith('.pdf'):
                src_path = os.path.join(sdir, fname)
                dest_path = os.path.join(OUTPUT_DIR, fname[:-4] + '.txt')
                
                if not os.path.exists(dest_path):
                    tasks.append((src_path, dest_path))

    print(f"Found {len(tasks)} PDFs to extract.")
    
    # Use ThreadPoolExecutor for I/O and PDF processing
    # pdfminer is CPU intensive but we can parallelize a bit
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_file, tasks))
    
    success_count = sum(1 for r in results if r)
    print(f"Finished! Successfully extracted {success_count} files.")

if __name__ == "__main__":
    main()
