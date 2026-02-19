import requests

url = "https://howdyportal.tamu.edu/uPortal/p/public-class-search-ui.ctf1/max/render.uP"
try:
    response = requests.get(url, timeout=10)
    with open("full_page.html", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Page saved.")
except Exception as e:
    print(f"Error: {e}")
