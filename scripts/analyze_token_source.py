import re

with open('class_search_component.js', 'r', encoding='utf-8') as f:
    content = f.read()

keywords = ['Bearer', 'token', 'jwt', 'sessionStorage', 'localStorage', 'cookie']

for kw in keywords:
    print(f"\n--- Context for '{kw}' ---")
    matches = re.finditer(re.escape(kw), content, re.IGNORECASE)
    for i, m in enumerate(matches):
        if i > 5:
            break  # limit output
        start = max(0, m.start() - 100)
        end = min(len(content), m.end() + 100)
        print(f"...{content[start:end]}...")

