import json
import time
import random
import requests
from pathlib import Path

METADATA_FILE = Path("dataset/wines_metadata.json")
BASE_API_URL = "https://vino-svoe.ru/api/wines/"

session = requests.Session()
session.headers.update({
    "accept": "application/json",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
})

# UPDATE THIS with your new cookie after switching networks
session.cookies.update({
    "cookiesession1": "678A3EB24DF0E98873F79DE78BBD9BD4", 
    "__svoe-vino-accepted-proof-of-age": "true"
})

def enrich_all_safely():
    if not METADATA_FILE.exists():
        print("[!] Metadata file not found.")
        return
        
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        wines = json.load(f)
        
    print(f"[*] Resuming metadata enrichment for {len(wines)} wines...")
    updated_wines = []
    processed_count = 0
    
    for i, wine_entry in enumerate(wines):
        raw_data = wine_entry.get("raw_metadata", {})
        slug = raw_data.get("slug")
        
        # 1. Instantly fix producer
        if not wine_entry.get("producer"):
            wine_entry["producer"] = raw_data.get("manufacturer") or raw_data.get("winery")
            
        # 2. Skip logic: If we already enriched this wine before the ban, skip API calls
        if slug and not (wine_entry.get("year") and wine_entry.get("rating_roskachestvo")):
            try:
                # Fetch Rating
                rating_res = session.get(f"{BASE_API_URL}{slug}/rating", timeout=10)
                if rating_res.status_code == 200:
                    r_data = rating_res.json()
                    wine_entry["rating_roskachestvo"] = r_data.get("rating_roskachestvo") or r_data.get("rating")
                
                # Fetch Main Details
                detail_res = session.get(f"{BASE_API_URL}{slug}", timeout=10)
                if detail_res.status_code == 200:
                    d_data = detail_res.json().get("data", detail_res.json())
                    if not wine_entry.get("year"):
                        wine_entry["year"] = d_data.get("year") or d_data.get("vintage")
                    if not wine_entry.get("grape"):
                        wine_entry["grape"] = d_data.get("grape") or d_data.get("varieties")
                    
                    wine_entry["detailed_metadata"] = d_data
                
                processed_count += 1
                
                # Add a random delay between 1.5 and 3.5 seconds to mimic human clicking
                time.sleep(random.uniform(1.5, 3.5))

            except Exception as e:
                print(f"[!] Network error on {slug}: {e}")
                
        updated_wines.append(wine_entry)
        
        # Save progress continually so you don't lose data if you get blocked again
        if (i + 1) % 10 == 0:
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(updated_wines, f, ensure_ascii=False, indent=2)
            print(f"Checked {i + 1}/{len(wines)}. API calls made this session: {processed_count}")

    # Final save
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_wines, f, ensure_ascii=False, indent=2)
        
    print("[+] Enrichment complete. Safe scraping successful.")

if __name__ == "__main__":
    enrich_all_safely()