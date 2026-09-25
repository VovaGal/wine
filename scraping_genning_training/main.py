import os
import time
import json
import requests
from pathlib import Path
from urllib.parse import urljoin

# Base configuration
BASE_URL = "https://vino-svoe.ru"
API_URL = f"{BASE_URL}/api/wines"
OUTPUT_DIR = Path("dataset")
IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_FILE = OUTPUT_DIR / "wines_metadata.json"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Headers and cookies translated directly from your cURL
HEADERS = {
    "accept": "*/*",
    "accept-language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "priority": "u=1, i",
    "referer": "https://vino-svoe.ru/wines",
    "sec-ch-ua": '"Chromium";v="152", "Not?A_Brand";v="24", "Google Chrome";v="152"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
}

COOKIES = {
    "cookiesession1": "678A3EB2D51D1791D18781E7A18DF983",
    "__svoe-vino-accepted-proof-of-age": "true"
}

def extract_image_url(item: dict) -> str | None:
    """Defensively checks common image keys in the wine object."""
    candidates = [
        item.get("image"),
        item.get("image_url"),
        item.get("imageUrl"),
        item.get("photo"),
        item.get("picture"),
        item.get("file", {}).get("url") if isinstance(item.get("file"), dict) else None,
    ]
    for url in candidates:
        if url and isinstance(url, str):
            return urljoin(BASE_URL, url)
    return None

def download_image(session: requests.Session, url: str, dest_path: Path) -> bool:
    """Downloads an image in chunks and skips if already present."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True

    try:
        with session.get(url, stream=True, timeout=15) as r:
            if r.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
    except Exception as e:
        print(f"  [!] Failed to download {url}: {e}")
    return False

def scrape_catalog(per_page: int = 50, delay: float = 0.4):
    """
    Iterates through all pages, extracts metadata, and saves images locally.
    Default per_page is increased to 50 to minimize total requests.
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    session.cookies.update(COOKIES)

    all_wines = []
    page = 1
    total_downloaded = 0

    print(f"[*] Starting scrape from {API_URL} (perPage={per_page})...")

    while True:
        params = {"page": page, "perPage": per_page}

        try:
            response = session.get(API_URL, params=params, timeout=15)
            
            # Handle rate-limiting or expired cookies
            if response.status_code in (401, 403):
                print(f"[!] Authorization/Age gate error (Status {response.status_code}). Refresh cookiesession1.")
                break
            elif response.status_code != 200:
                print(f"[!] Non-200 response ({response.status_code}) on page {page}. Stopping.")
                print(f"[!] Server says: {response.text}")
                break

            payload = response.json()

            # Handle either list response or dict wrapper (items / data / wines)
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, dict):
                items = (
                    payload.get("items")
                    or payload.get("data")
                    or payload.get("wines")
                    or payload.get("results")
                    or []
                )
            else:
                items = []

            if not items:
                print(f"[*] No items found on page {page}. Reached end of catalog.")
                break

            print(f"[*] Page {page}: processing {len(items)} wines...")

            for item in items:
                wine_id = str(item.get("id") or item.get("code") or len(all_wines) + 1)
                img_url = extract_image_url(item)

                local_img_path = None
                if img_url:
                    ext = Path(img_url.split("?")[0]).suffix or ".png"
                    filename = f"wine_{wine_id}{ext}"
                    dest = IMAGES_DIR / filename
                    
                    if download_image(session, img_url, dest):
                        local_img_path = str(dest)
                        total_downloaded += 1

                # Retain all original attributes plus resolved local path
                item_record = {
                    "scraped_id": wine_id,
                    "name": item.get("name") or item.get("title"),
                    "producer": item.get("producer") or item.get("winery"),
                    "year": item.get("year") or item.get("vintage"),
                    "color": item.get("color"),
                    "sugar": item.get("sugar") or item.get("category"),
                    "region": item.get("region"),
                    "grape": item.get("grape") or item.get("varieties"),
                    "rating_roskachestvo": item.get("rating_roskachestvo") or item.get("rating"),
                    "remote_image_url": img_url,
                    "local_image_path": local_img_path,
                    "raw_metadata": item
                }
                all_wines.append(item_record)

            # Checkpoint metadata to disk on every page in case of network drops
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(all_wines, f, ensure_ascii=False, indent=2)

            page += 1
            time.sleep(delay)

        except requests.exceptions.RequestException as e:
            print(f"[!] Network error on page {page}: {e}. Retrying after 2 seconds...")
            time.sleep(2.0)

    print(f"\n[+] Scraping complete.")
    print(f"[+] Total wines indexed: {len(all_wines)}")
    print(f"[+] Images saved: {total_downloaded}")
    print(f"[+] Metadata written to: {METADATA_FILE.resolve()}")

if __name__ == "__main__":
    scrape_catalog(per_page=16, delay=0.4)