import json
from pathlib import Path
from urllib.parse import urljoin
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://api.vino-svoe.ru"
METADATA_FILE = Path("dataset/wines_metadata.json")
IMAGES_DIR = Path("dataset/images")

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# 1. Setup a persistent session with the required anti-bot headers
session = requests.Session()
session.headers.update({
    "accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "accept-language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "referer": "https://vino-svoe.ru/wines",
    "sec-ch-ua": '"Chromium";v="152", "Not?A_Brand";v="24", "Google Chrome";v="152"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
})
session.cookies.update({
    "cookiesession1": "678A3EB2D51D1791D18781E7A18DF983",
    "__svoe-vino-accepted-proof-of-age": "true"
})

# 2. Add automatic retries for dropped connections
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[403, 429, 500, 502, 503, 504],
)
session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

def extract_url_defensively(raw_data):
    """Checks multiple possible keys for the image URL."""
    # Check standard nested dictionary: "image": {"url": "..."}
    if isinstance(raw_data.get("image"), dict) and raw_data["image"].get("url"):
        return raw_data["image"]["url"]
    
    # Check flat strings just in case
    for key in ["image", "imageUrl", "preview_picture", "picture"]:
        if isinstance(raw_data.get(key), str):
            return raw_data[key]
            
    return None

def download_single_image(wine_entry):
    wine_id = wine_entry["scraped_id"]
    raw_data = wine_entry.get("raw_metadata", {})
    
    img_url_partial = extract_url_defensively(raw_data)
    
    if not img_url_partial:
        # We silently return True here so it doesn't count as an error, 
        # it just means the database entry literally has no picture.
        return wine_entry, False, "No URL in JSON"

    #img_url = urljoin(BASE_URL, img_url_partial)
    # Strip the leading slash from the JSON path so it appends cleanly
    clean_path = img_url_partial.lstrip('/')
    
    # Force the server to generate a high-resolution 800x800 image
    img_url = f"{BASE_URL}/v1/img/str-api/800/800/resize/{clean_path}"
    wine_entry["remote_image_url"] = img_url

    ext = Path(img_url.split("?")[0]).suffix or ".webp"
    dest_path = IMAGES_DIR / f"wine_{wine_id}{ext}"
    
    if dest_path.exists() and dest_path.stat().st_size > 0:
        wine_entry["local_image_path"] = str(dest_path)
        return wine_entry, True, "Already downloaded"
        
    try:
        response = session.get(img_url, timeout=15)
        if response.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(response.content)
            wine_entry["local_image_path"] = str(dest_path)
            return wine_entry, True, "Success"
        else:
            return wine_entry, False, f"HTTP {response.status_code}"
    except Exception as e:
        return wine_entry, False, f"Error: {str(e)[:50]}"

def download_all_images():
    if not METADATA_FILE.exists():
        print("[!] Metadata file not found.")
        return
        
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        wines = json.load(f)
        
    print(f"[*] Found {len(wines)} wines. Starting downloads...")
    updated_wines = []
    success_count = 0
    missing_count = 0
    
    # Reduced max_workers to 4 to prevent triggering connection drops from the firewall
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(download_single_image, wines)
        
        for i, (updated_entry, success, msg) in enumerate(results):
            updated_wines.append(updated_entry)
            if success:
                if msg == "Success":
                    success_count += 1
            else:
                if msg == "No URL in JSON":
                    missing_count += 1
                else:
                    print(f"  [!] ID {updated_entry.get('scraped_id')} failed: {msg}")
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(wines)} | Downloaded: {success_count} | No Image: {missing_count}")
                
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_wines, f, ensure_ascii=False, indent=2)
        
    print(f"\n[+] Task complete!")
    print(f"[+] Newly downloaded: {success_count}")
    print(f"[+] Entries with no image available: {missing_count}")

if __name__ == "__main__":
    download_all_images()