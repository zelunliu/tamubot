
import requests

cookies = {
    'howdyportal': '036ddbae596d765f',
    'JSESSIONID': '959F9AC2247889488E0E21CBE73D2CDD',
    'org.apereo.portal.PORTLET_COOKIE': 'uJFZhxNpKoleq78jKNAd2c5L5CoLpmunHlU_dl13'
}

base_url = "https://howdyportal.tamu.edu"
api_url = f"{base_url}/uPortal/api/v5-1/userinfo"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

print(f"Testing {api_url}...")
try:
    response = requests.get(api_url, cookies=cookies, headers=headers, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Body: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
