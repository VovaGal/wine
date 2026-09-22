import urllib.request
from pathlib import Path
import time

# Target directory
FONTS_DIR = Path("dataset/ocr_training/fonts")
FONTS_DIR.mkdir(parents=True, exist_ok=True)

# Direct links to the raw .ttf files in Google's official GitHub repository
# This bypasses the Google Fonts website bot-protection entirely.
DIRECT_TTF_URLS = [
    # --- Handwritten / Script / Fancy (Full Unicode) ---
    "https://raw.githubusercontent.com/google/fonts/main/ofl/lobster/Lobster-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/marckscript/MarckScript-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/badscript/BadScript-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/neucha/Neucha.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/kurale/Kurale-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/forum/Forum-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/yesevaone/YesevaOne-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/kellyslab/KellySlab-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/philosopher/Philosopher-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/philosopher/Philosopher-Italic.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/alice/Alice-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/prata/Prata-Regular.ttf",
    
    # --- Classic Serif (Essential for French Wine Labels) ---
    
    # --- Clean Sans-Serif (Modern Labels) ---
    "https://raw.githubusercontent.com/google/fonts/main/ofl/opensans/static/OpenSans-Regular.ttf"

]

def download_raw_fonts():
    print(f"[*] Downloading {len(DIRECT_TTF_URLS)} raw fonts from GitHub into {FONTS_DIR}...")
    
    for url in DIRECT_TTF_URLS:
        filename = url.split("/")[-1]
        target_path = FONTS_DIR / filename
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                with open(target_path, "wb") as f:
                    f.write(response.read())
            print(f"[+] Downloaded: {filename}")
        except Exception as e:
            print(f"[-] Failed to download {filename}: {e}")
            
        time.sleep(0.2)

    print(f"\n[*] Done! Your fonts are ready in {FONTS_DIR.absolute()}")

if __name__ == "__main__":
    download_raw_fonts()