
import requests

cookies = {
    'howdyportal': '036ddbae596d765f',
    'JSESSIONID': '959F9AC2247889488E0E21CBE73D2CDD',
    'org.apereo.portal.PORTLET_COOKIE': 'uJFZhxNpKoleq78jKNAd2c5L5CoLpmunHlU_dl13'
}

base_url = "https://howdyportal.tamu.edu"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# 1. Get Token
print("Getting token...")
token_resp = requests.get(f"{base_url}/uPortal/api/v5-1/userinfo", cookies=cookies, headers=headers)
token = token_resp.text.strip()
print(f"Token received: {token[:20]}...")

# 2. Fetch Syllabus PDF
syllabus_url = f"{base_url}/api/course-syllabus-pdf"
payload = {
    "termCode": "202411",
    "crn": "50142" 
}
auth_headers = headers.copy()
auth_headers['Authorization'] = f"Bearer {token}"
auth_headers['Content-Type'] = 'application/json'

print(f"Fetching syllabus for {payload}...")
pdf_resp = requests.post(syllabus_url, json=payload, headers=auth_headers, cookies=cookies)

print(f"Status: {pdf_resp.status_code}")
print(f"Content-Type: {pdf_resp.headers.get('Content-Type')}")

if pdf_resp.status_code == 200 and 'application/pdf' in pdf_resp.headers.get('Content-Type', ''):
    with open("syllabus_check.pdf", "wb") as f:
        f.write(pdf_resp.content)
    print("Syllabus saved as syllabus_check.pdf")
else:
    print("Failed to download PDF.")
    print("Response:", pdf_resp.text[:200])
