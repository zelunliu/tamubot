import os
import json
from collections import Counter

def analyze_chunks():
    source_dir = 'tamu_data/tamu_scraper/syllabi'
    output_dir = 'tamu_data/rag_vertex/styled_chunks'
    
    source_files = {f.replace('.pdf', '') for f in os.listdir(source_dir) if f.endswith('.pdf')}
    output_files = {f.replace('.json', '') for f in os.listdir(output_dir) if f.endswith('.json')}
    
    failed_files = sorted(list(source_files - output_files))
    
    header_counts = Counter()
    total_chunks = 0
    empty_files = []
    
    processed_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    
    for fname in processed_files:
        path = os.path.join(output_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not data:
                    empty_files.append(fname.replace('.json', ''))
                    continue
                
                total_chunks += len(data)
                for chunk in data:
                    header_counts[chunk.get('header', 'Unknown').strip()] += 1
        except:
            continue

    stats = {
        "total_source_files": len(source_files),
        "successfully_processed": len(output_files) - len(empty_files),
        "failed_to_extract": len(failed_files),
        "extracted_empty": len(empty_files),
        "total_chunks_created": total_chunks,
        "failed_list": failed_files + empty_files
    }
    
    with open('chunking_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
        
    print(f"Total Source: {stats['total_source_files']}")
    print(f"Success: {stats['successfully_processed']}")
    print(f"Failed (Missing): {stats['failed_to_extract']}")
    print(f"Failed (Empty): {stats['extracted_empty']}")
    
    print("\nTop 50 Most Common Headers:")
    for h, c in header_counts.most_common(50):
        print(f"{c:5d} | {h}")

if __name__ == "__main__":
    analyze_chunks()
