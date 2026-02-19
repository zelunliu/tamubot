
import os
import json
import statistics
from collections import defaultdict

def analyze_dept_quality():
    input_dir = 'tamu_data/rag_vertex/standardized_chunks'
    dept_stats = defaultdict(lambda: {"total": 0, "mapped": 0, "lengths": []})
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for fname in files:
        # Extract dept from filename: 202611_CSCE_... -> CSCE
        parts = fname.split('_')
        if len(parts) < 2: continue
        dept = parts[1]
        
        path = os.path.join(input_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                
            for chunk in chunks:
                dept_stats[dept]["total"] += 1
                if chunk["standard_header"] != "MISC_OR_UNKNOWN":
                    dept_stats[dept]["mapped"] += 1
                    # Measure length of the content
                    length = len(chunk.get("content", ""))
                    if length > 0:
                        dept_stats[dept]["lengths"].append(length)
        except:
            continue

    # Process results
    results = []
    for dept, data in dept_stats.items():
        if data["total"] == 0: continue
        
        efficiency = (data["mapped"] / data["total"]) * 100
        
        if len(data["lengths"]) > 1:
            avg_len = statistics.mean(data["lengths"])
            std_dev = statistics.stdev(data["lengths"])
        elif len(data["lengths"]) == 1:
            avg_len = data["lengths"][0]
            std_dev = 0
        else:
            avg_len = 0
            std_dev = 0
            
        results.append({
            "dept": dept,
            "efficiency": efficiency,
            "avg_len": avg_len,
            "std_dev": std_dev,
            "sample_size": data["total"]
        })

    # Sort by efficiency (Best headers)
    results.sort(key=lambda x: x["efficiency"], reverse=True)
    
    # Save full results
    with open('dept_header_quality.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Print Top 15
    print(f"{'DEPT':<6} | {'EFFICIENCY':<10} | {'AVG LEN':<10} | {'STD DEV':<10} | {'CHUNKS'}")
    print("-" * 60)
    for r in results[:15]:
        print(f"{r['dept']:<6} | {r['efficiency']:>9.2f}% | {r['avg_len']:>10.1f} | {r['std_dev']:>10.1f} | {r['sample_size']}")

if __name__ == "__main__":
    analyze_dept_quality()
