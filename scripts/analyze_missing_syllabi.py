import json
from collections import defaultdict

total_sections = 0
total_no_syl = 0
dept_no_syl = defaultdict(list)
targets = {'ISEN', 'STAT', 'CSCE'}

with open('tamu_data/tamu_scraper/spring2026_data.jsonl', 'r') as f:
    for line in f:
        total_sections += 1
        data = json.loads(line)
        has_syl = data.get('raw_data', {}).get('SWV_CLASS_SEARCH_HAS_SYL_IND')
        if has_syl == 'N':
            total_no_syl += 1
            subj = data.get('subject')
            if subj in targets:
                course_info = f"{data.get('subject')} {data.get('course')} (Section {data.get('section')}) - {data.get('title')}"
                dept_no_syl[subj].append(course_info)

print(f"Total sections checked: {total_sections}")
print(f"Total sections without syllabus: {total_no_syl}")
print(f"Percentage missing: {(total_no_syl/total_sections)*100:.2f}%")

for dept in sorted(targets):
    sections = dept_no_syl[dept]
    print(f"\n--- {dept} ({len(sections)} sections missing syllabi) ---")
    courses = defaultdict(list)
    for s in sections:
        parts = s.split(' ')
        if len(parts) > 1:
            num = parts[1]
            courses[num].append(s)
    
    for num in sorted(courses.keys()):
        print(f"Course {num}: {len(courses[num])} section(s) missing")
        for s in sorted(courses[num]):
            print(f"  - {s}")
