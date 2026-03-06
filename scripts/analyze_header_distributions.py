
import json
import os
import statistics
from collections import defaultdict


def analyze_header_distributions():
    input_dir = 'tamu_data/rag_vertex/standardized_chunks'
    category_stats = defaultdict(list)
    total_chunks = 0
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for fname in files:
        path = os.path.join(input_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                
            for chunk in chunks:
                total_chunks += 1
                cat = chunk.get("standard_header", "MISC_OR_UNKNOWN")
                length = len(chunk.get("content", ""))
                if length > 0:
                    category_stats[cat].append(length)
        except Exception:
            continue

    results = []
    for cat, lengths in category_stats.items():
        count = len(lengths)
        if count > 0:
            avg = statistics.mean(lengths)
            std = statistics.stdev(lengths) if count > 1 else 0
            median = statistics.median(lengths)
            results.append({
                "category": cat,
                "count": count,
                "proportion": (count / total_chunks) * 100,
                "avg": avg,
                "std": std,
                "median": median,
                "min": min(lengths),
                "max": max(lengths)
            })

    # Sort by count (proportions)
    results.sort(key=lambda x: x["count"], reverse=True)
    
    with open('header_distribution_stats.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Print Table
    print(f"{'CATEGORY':<20} | {'PROP %':<7} | {'AVG':<8} | {'MEDIAN':<8} | {'STD':<8} | {'COUNT'}")
    print("-" * 75)
    for r in results:
        print(f"{r['category']:<20} | {r['proportion']:>6.1f}% | {r['avg']:>8.1f} | {r['median']:>8.1f} | {r['std']:>8.1f} | {r['count']}")

if __name__ == "__main__":
    analyze_header_distributions()
