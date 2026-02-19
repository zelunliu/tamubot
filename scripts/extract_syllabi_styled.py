import fitz  # PyMuPDF
import os
import json
import re
from collections import Counter
import concurrent.futures

def clean_content(text):
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"Course Syllabus", "", text)
    lines = [line.strip() for line in text.split('\n')]
    return "\n".join([l for l in lines if l])

def extract_course_info(doc):
    first_page_text = doc[0].get_text()
    match = re.search(r"([A-Z]{4}\s+\d{3})", first_page_text)
    return match.group(1) if match else "Unknown Course"

def process_single_pdf(args):
    fname, source_dir, output_dir = args
    pdf_path = os.path.join(source_dir, fname)
    output_path = os.path.join(output_dir, fname.replace('.pdf', '.json'))
    
    if os.path.exists(output_path):
        return False # Skipped
        
    try:
        doc = fitz.open(pdf_path)
        course_id = extract_course_info(doc)
        
        font_sizes = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            font_sizes.append(round(s["size"]))
        
        if not font_sizes:
            return False
            
        common_size = Counter(font_sizes).most_common(1)[0][0]
        
        chunks = []
        current_chunk = {"course": course_id, "header": "Introduction", "content": []}
        
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    block_text = ""
                    is_header = False
                    
                    for l in b["lines"]:
                        line_text = ""
                        for s in l["spans"]:
                            if round(s["size"]) > common_size or (s["flags"] & 2**4):
                                is_header = True
                            line_text += s["text"]
                        block_text += line_text + " "
                    
                    block_text = block_text.strip()
                    if not block_text or block_text == "Course Syllabus":
                        continue
                        
                    if is_header and len(block_text) < 100:
                        if current_chunk["content"]:
                            current_chunk["content"] = clean_content("\n".join(current_chunk["content"]))
                            chunks.append(current_chunk)
                        current_chunk = {"course": course_id, "header": block_text, "content": []}
                    else:
                        current_chunk["content"].append(block_text)
        
        if current_chunk["content"]:
            if isinstance(current_chunk["content"], list):
                current_chunk["content"] = clean_content("\n".join(current_chunk["content"]))
            chunks.append(current_chunk)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
        return True
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return False

def main():
    source_dir = 'tamu_data/tamu_scraper/syllabi'
    output_dir = 'tamu_data/rag_vertex/styled_chunks'
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]
    print(f"Total PDFs found: {len(files)}")
    
    tasks = [(f, source_dir, output_dir) for f in files]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_pdf, tasks))
    
    processed_count = sum(1 for r in results if r)
    print(f"Successfully processed {processed_count} new syllabi.")

if __name__ == "__main__":
    main()
