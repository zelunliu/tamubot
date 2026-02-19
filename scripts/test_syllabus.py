import requests
import json

cookies = {
    'howdyportal': '036ddbae596d765f',
    'JSESSIONID': '959F9AC2247889488E0E21CBE73D2CDD',
    'org.apereo.portal.PORTLET_COOKIE': 'uJFZhxNpKoleq78jKNAd2c5L5CoLpmunHlU_dl13'
}

base_url = "https://howdyportal.tamu.edu"
# I recall seeing this endpoint
api_url = f"{base_url}/api/course-syllabus-pdf"

headers = {
    'Content-Type': 'application/json; charset=utf-8',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Guessing the payload based on other endpoints. 
# Usually needs termCode and crn.
payload = {
    "termCode": "202411",
    "crn": "50142" 
}

print(f"Testing syllabus download for {payload}...")
try:
    response = requests.post(api_url, json=payload, cookies=cookies, headers=headers, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    print(f"Content-Length: {response.headers.get('Content-Length')}")
    
    if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
        with open("test_syllabus.pdf", "wb") as f:
            f.write(response.content)
        print("Saved test_syllabus.pdf")
    else:
        print("Response text preview:", response.text[:200])

except Exception as e:
    print(f"Error: {e}")
