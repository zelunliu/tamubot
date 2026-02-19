import requests

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

base_url = "https://howdyportal.tamu.edu"

# 1. Visit main page to set cookies
print("Visiting main page...")
main_url = f"{base_url}/uPortal/p/public-class-search-ui.ctf1/max/render.uP"
session.get(main_url)

# 2. Get Token
print("Getting token...")
token_resp = session.get(f"{base_url}/uPortal/api/v5-1/userinfo")
token = token_resp.text.strip()
print(f"Token received: {token[:20]}...")

# 3. Fetch Syllabus PDF
syllabus_url = f"{base_url}/api/course-syllabus-pdf"
payload = {
    "termCode": "202411",
    "crn": "50142" 
}
headers = {
    'Authorization': f"Bearer {token}",
    'Content-Type': 'application/json'
}

print(f"Fetching syllabus for {payload}...")
pdf_resp = session.post(syllabus_url, json=payload, headers=headers)

print(f"Status: {pdf_resp.status_code}")
print(f"Content-Type: {pdf_resp.headers.get('Content-Type')}")

if pdf_resp.status_code == 200 and 'application/pdf' in pdf_resp.headers.get('Content-Type', ''):
    with open("syllabus_check_session.pdf", "wb") as f:
        f.write(pdf_resp.content)
    print("Syllabus saved as syllabus_check_session.pdf")
else:
    print("Failed to download PDF.")
    print("Response:", pdf_resp.text[:200])
