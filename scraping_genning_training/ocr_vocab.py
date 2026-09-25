import json
import re
from pathlib import Path
from collections import Counter

METADATA_FILE = Path("dataset/wines_metadata.json")
OUTPUT_DIR = Path("dataset/ocr_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    """Removes excessive whitespace and standardizes casing."""
    if not isinstance(text, str):
        return ""
    # Keep Cyrillic, Latin, numbers, and basic punctuation (hyphens, percentages)
    text = re.sub(r'[^\w\s\-\.%]', '', text)
    return " ".join(text.split()).strip()

def build_vocabulary():
    if not METADATA_FILE.exists():
        print("[!] Metadata file not found.")
        return
        
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        wines = json.load(f)
        
    phrases = set()
    words = []
    
    print(f"[*] Extracting vocabulary from {len(wines)} wines...")
    
    for wine in wines:
        detailed = wine.get("detailed_metadata") or {}
        
        # 1. Grab top-level fields
        fields = [
            wine.get("producer"),
            wine.get("name"),
            wine.get("region"),
            wine.get("sugar"),
            wine.get("color")
        ]
        
        # 2. Extract Grapes from the nested dictionary
        grapes_list = detailed.get("grapes", [])
        for grape_obj in grapes_list:
            if isinstance(grape_obj, dict) and grape_obj.get("name"):
                fields.append(grape_obj.get("name"))
                
        # 3. Extract Year from the nested dictionary
        year = str(detailed.get("year", "")) or str(detailed.get("vintage", ""))
        if year and year.isdigit():
            phrases.add(year)
            phrases.add(f"{year}г.")
            
        # 4. Clean and add to lexicon
        for field in fields:
            cleaned = clean_text(field)
            if cleaned:
                # Add the exact phrase
                phrases.add(cleaned)
                
                # Break down into individual words
                for word in cleaned.split():
                    if len(word) > 1 or word.isdigit():
                        words.append(word)

    # Filter and sort words by frequency
    word_counts = Counter(words)
    common_words = [word for word, count in word_counts.items() if count >= 1]
    
    # Combine phrases and unique words
    master_lexicon = sorted(list(phrases.union(set(common_words))))
    
    # Remove empty strings
    master_lexicon = [item for item in master_lexicon if item]
    
    vocab_file = OUTPUT_DIR / "wine_lexicon.txt"
    with open(vocab_file, "w", encoding="utf-8") as f:
        for item in master_lexicon:
            f.write(f"{item}\n")
            
    print(f"[+] Extracted {len(master_lexicon)} unique domain terms.")
    print(f"[+] Lexicon saved to: {vocab_file}")

if __name__ == "__main__":
    build_vocabulary()