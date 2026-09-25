import json
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

METADATA_FILE = Path("dataset/wines_metadata.json")

def convert_single_image(wine_entry):
    local_path = wine_entry.get("local_image_path")
    
    if not local_path or not local_path.endswith(".webp"):
        return wine_entry, False
        
    webp_file = Path(local_path)
    if not webp_file.exists():
        return wine_entry, False
        
    png_file = webp_file.with_suffix(".png")
    
    try:
        # Convert to PNG, retaining the alpha transparency channel (RGBA)
        with Image.open(webp_file) as img:
            img.convert("RGBA").save(png_file, "PNG")
            
        # Update JSON to point to the new PNG file
        wine_entry["local_image_path"] = str(png_file)
        
        # Delete the original WebP to save space
        webp_file.unlink()
        return wine_entry, True
        
    except Exception as e:
        print(f"[!] Error converting {webp_file.name}: {e}")
        return wine_entry, False

def convert_all():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        wines = json.load(f)
        
    print(f"[*] Starting conversion of {len(wines)} images to PNG...")
    updated_wines = []
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(convert_single_image, wines)
        
        for i, (updated_entry, success) in enumerate(results):
            updated_wines.append(updated_entry)
            if success:
                success_count += 1
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(wines)}... (Converted: {success_count})")
                
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_wines, f, ensure_ascii=False, indent=2)
        
    print(f"\n[+] Conversion complete! {success_count} images are now PNGs.")

if __name__ == "__main__":
    convert_all()