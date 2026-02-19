import requests

url = "https://cdn.eis.tamu.edu/webcomponents/prod/class-search-ui-element.js"
try:
    response = requests.get(url, timeout=10)
    with open("class_search_component.js", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("JS saved.")
except Exception as e:
    print(f"Error: {e}")
