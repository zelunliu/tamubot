import re

with open('class_search_component.js', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = "course-syllabus-pdf"
matches = re.finditer(pattern, content)

for m in matches:
    start = max(0, m.start() - 200)
    end = min(len(content), m.end() + 200)
    print(f"\n--- Context ---")
    print(content[start:end])
