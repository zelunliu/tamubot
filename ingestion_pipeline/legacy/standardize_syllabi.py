
import concurrent.futures
import json
import os
import re
from collections import Counter

import fitz

# 1. Definitive Standard Mapping
HEADER_MAPPING = {
    "COURSE_INFO": [r"course info", r"catalog description", r"course description", r"special course designation", r"syllabus", r"course details"],
    "INSTRUCTOR_INFO": [r"instructor detail", r"teaching assistant", r"contact info", r"office hours"],
    "PREREQUISITES": [r"prerequisite"],
    "LEARNING_OUTCOMES": [r"learning outcome", r"learning activit", r"learning objective", r"learner will be able to"],
    "MATERIALS": [r"textbook", r"resource material", r"learning resource", r"required device", r"calculator policy", r"electronic devices"],
    "GRADING": [r"grading policy", r"grading scale", r"assignments", r"quizzes", r"exams", r"homework", r"total points", r"grade appeals"],
    "ATTENDANCE": [r"attendance policy", r"attendance"],
    "LATE_WORK": [r"late work", r"makeup work", r"make-up"],
    "SCHEDULE": [r"course schedule", r"calendar", r"weekly", r"week \d+", r"march", r"january", r"february", r"april"],
    "UNIVERSITY_POLICIES": [r"university polic", r"ada policy", r"academic integrity", r"nondiscrimination", r"mental health", r"wellness", r"ferpa", r"title ix", r"pregnancy", r"civil rights", r"free speech", r"disabilities act", r"copyright"],
    "SUPPORT_SERVICES": [r"technology support", r"it service", r"canvas support", r"help desk", r"learning center", r"help session"],
    "AI_POLICY": [r"ai statement", r"artificial intelligence", r"outside resources"]
}

def get_standard_header(raw_header):
    raw_lower = raw_header.lower().strip()
    for category, patterns in HEADER_MAPPING.items():
        for pattern in patterns:
            if re.search(pattern, raw_lower):
                return category
    return "MISC_OR_UNKNOWN"

def clean_content(text):
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"Course Syllabus", "", text)
    lines = [line.strip() for line in text.split('\n')]
    return "\n".join([ln for ln in lines if ln])

def extract_course_info(doc):
    first_page_text = doc[0].get_text()
    match = re.search(r"([A-Z]{4}\s+\d{3})", first_page_text)
    return match.group(1) if match else "Unknown Course"

def process_single_pdf(args):
    fname, source_dir, output_dir = args
    pdf_path = os.path.join(source_dir, fname)
    output_path = os.path.join(output_dir, fname.replace('.pdf', '.json'))
    
    try:
        doc = fitz.open(pdf_path)
        course_id = extract_course_info(doc)
        
        font_sizes = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for ln in b["lines"]:
                        for s in ln["spans"]:
                            font_sizes.append(round(s["size"]))
        
        if not font_sizes:
            return None
        common_size = Counter(font_sizes).most_common(1)[0][0]
        
        chunks = []
        current_chunk = {"course": course_id, "standard_header": "COURSE_INFO", "raw_header": "Introduction", "content": []}
        
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    block_text = "".join([s["text"] for ln in b["lines"] for s in ln["spans"]]).strip()
                    if not block_text or block_text == "Course Syllabus":
                        continue
                    is_styled = any(round(s["size"]) > common_size or (s["flags"] & 2**4) for ln in b["lines"] for s in ln["spans"])
                    
                    if is_styled and len(block_text) < 80:
                        std_h = get_standard_header(block_text)
                        
                        if current_chunk["content"]:
                            current_chunk["content"] = clean_content("\n".join(current_chunk["content"]))
                            chunks.append(current_chunk)
                        
                        current_chunk = {
                            "course": course_id,
                            "standard_header": std_h,
                            "raw_header": block_text,
                            "content": []
                        }
                    else:
                        current_chunk["content"].append(block_text)
        
        if current_chunk["content"]:
            current_chunk["content"] = clean_content("\n".join(current_chunk["content"]))
            chunks.append(current_chunk)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
        return chunks
    except Exception:
        return None

def main():
    source_dir = 'tamu_data/tamu_scraper/syllabi'
    output_dir = 'tamu_data/rag_vertex/standardized_chunks'
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]
    tasks = [(f, source_dir, output_dir) for f in files]
    
    print(f"Reprocessing {len(files)} files with refined mapping...")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_pdf, tasks))
    
    unmapped_headers = Counter()
    mapped_count = 0
    total_chunks = 0
    
    for result in results:
        if result:
            for chunk in result:
                total_chunks += 1
                if chunk["standard_header"] == "MISC_OR_UNKNOWN":
                    unmapped_headers[chunk["raw_header"]] += 1
                else:
                    mapped_count += 1
    
    report = {
        "summary": {
            "total_files": len(files),
            "total_chunks": total_chunks,
            "mapped_chunks": mapped_count,
            "unmapped_chunks": total_chunks - mapped_count,
            "mapping_efficiency": f"{(mapped_count/total_chunks)*100:.2f}%" if total_chunks > 0 else "0%"
        },
        "top_unmapped": unmapped_headers.most_common(50)
    }
    
    with open('final_standardization_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"Standardization Finalized. Efficiency: {report['summary']['mapping_efficiency']}")

if __name__ == "__main__":
    main()
