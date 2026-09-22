import json
import re
import random
from pathlib import Path

def is_cyrillic(text: str) -> bool:
    """Classifies a string as Cyrillic if it contains Russian characters."""
    return bool(re.search(r'[А-Яа-яЁё]', text))

def augment_dataset(json_file: str, output_file: str):
    script_dir = Path(__file__).parent
    input_path = script_dir / "dataset" / json_file
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find {json_file}")

    print(f"[*] Parsing JSON metadata from {json_file}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        wines = json.load(f)

    # 1. Extract label-specific terminology into a set to remove duplicates
    raw_phrases = set()
    for wine in wines:
        if wine.get("name"): raw_phrases.add(wine["name"])
        if wine.get("producer"): raw_phrases.add(wine["producer"])
        if wine.get("region"): raw_phrases.add(wine["region"])
        if wine.get("sugar"): raw_phrases.add(wine["sugar"])
        
        # Dig into detailed_metadata for grape varieties
        detailed = wine.get("detailed_metadata", {})
        for grape in detailed.get("grapes", []):
            if grape.get("name"): raw_phrases.add(grape["name"])

    cyrillic_lines = []
    latin_lines = []

    # 2. Categorize the extracted phrases
    for phrase in raw_phrases:
        cleaned = phrase.strip()
        if not cleaned: 
            continue
        if is_cyrillic(cleaned):
            cyrillic_lines.append(cleaned)
        else:
            latin_lines.append(cleaned)

    print(f"[*] Extracted {len(raw_phrases)} unique phrases from JSON.")
    print(f"    Cyrillic phrases: {len(cyrillic_lines)}")
    print(f"    Latin phrases:    {len(latin_lines)}")

    # 3. Inject International Domain Corpus
    generic_latin_terms = [
        "Chianti", "Riserva", "Denominazione di Origine Controllata", "Garantita",
        "Cabernet Sauvignon", "Château", "Merlot", "Pinot Noir", "Chardonnay",
        "Sauvignon Blanc", "Syrah", "Shiraz", "Bordeaux", "Burgundy", "Prosecco",
        "Champagne", "Brut", "Blanc de Blancs", "Cuvée", "Grand Cru", "Premier Cru",
        "Sangiovese", "Tempranillo", "Rioja", "Moscato", "Zinfandel", "Malbec",
        "Vendemmia", "Reserva", "Gran Reserva", "Vin de Pays", "Appellation",
        "Estate Bottled", "Mis en bouteille au château", "Product of Italy",
        "Vino Tinto", "Vino Blanco", "Rosé", "Toscana", "Piemonte", "Veneto"
    ]
    generic_latin_terms.extend([
        # Inject standalone years
        *[str(year) for year in range(1990, 2026)],
        # Inject standard volumes and ABV
        "750ml", "750 ml", "75 cl", "1.5L", "12.5%", "13%", "13.5% vol", "14%"
    ])
    latin_lines.extend(generic_latin_terms)
    

    # 4. Frequency Balancing (Upsampling Latin to match Cyrillic)
    target_count = len(cyrillic_lines)
    current_latin_count = len(latin_lines)
    
    if current_latin_count < target_count:
        multiplier = target_count // current_latin_count
        remainder = target_count % current_latin_count
        balanced_latin = latin_lines * multiplier + random.sample(latin_lines, remainder)
    else:
        balanced_latin = random.sample(latin_lines, target_count)

    # 5. Final Aggregation and Shuffle
    final_lexicon = cyrillic_lines + balanced_latin
    random.shuffle(final_lexicon)

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in final_lexicon:
            f.write(f"{line}\n")

    print(f"[*] Augmented dataset distribution:")
    print(f"    Total Cyrillic: {len(cyrillic_lines)}")
    print(f"    Total Latin:    {len(balanced_latin)}")
    print(f"[*] Ready for TRDG! Balanced lexicon saved to {output_file}")

if __name__ == "__main__":
    augment_dataset("wines_metadata.json", "balanced_wines_lexicon.txt")