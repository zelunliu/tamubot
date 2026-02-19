import json
import os

JSONL_FILE = 'tamu_data/tamu_scraper/spring2026_data.jsonl'
SYLLABI_DIR = 'tamu_data/tamu_scraper/syllabi'

expected = set()
with open(JSONL_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data.get('raw_data', {}).get('SWV_CLASS_SEARCH_HAS_SYL_IND') == 'Y':
            term = data.get('term_code')
            subj = data.get('subject')
            num = data.get('course')
            sec = data.get('section')
            crn = data.get('crn')
            filename = f"{term}_{subj}_{num}_{sec}_{crn}.pdf"
            expected.add(filename)

print(f"Expected: {len(expected)}")

actual = set(f for f in os.listdir(SYLLABI_DIR) if f.endswith('.pdf'))
print(f"Actual: {len(actual)}")

missing = expected - actual
print(f"Missing ({len(missing)}):")
for m in missing:
    print(m)
