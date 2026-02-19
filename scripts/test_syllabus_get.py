import requests

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

base_url = "https://howdyportal.tamu.edu"

# 1. Visit main page to set cookies
print("Visiting main page...")
session.get(f"{base_url}/uPortal/p/public-class-search-ui.ctf1/max/render.uP")

# 2. Fetch Syllabus PDF via GET
syllabus_url = f"{base_url}/api/course-syllabus-pdf"
params = {
    "termCode": "202411",
    "crn": "50142" 
}

print(f"Fetching syllabus from {syllabus_url} with params {params}...")
pdf_resp = session.get(syllabus_url, params=params)

print(f"Status: {pdf_resp.status_code}")
print(f"Content-Type: {pdf_resp.headers.get('Content-Type')}")

if pdf_resp.status_code == 200 and 'application/pdf' in pdf_resp.headers.get('Content-Type', ''):
    with open("syllabus_get.pdf", "wb") as f:
        f.write(pdf_resp.content)
    print("Syllabus saved as syllabus_get.pdf")
else:
    print("Failed to download PDF.")
    print("Response:", pdf_resp.text[:200])
