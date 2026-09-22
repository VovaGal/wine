"""Build OCR lexicons from the scraped wine catalogue.

The output is deliberately split into language/script pools. Do sampling in
the synthetic-data generator; do not balance a lexicon by duplicating lines.

Example:
    python lexicon_upgrade.py --input dataset/wines_metadata.json \
        --output-dir dataset/ocr_training/lexicons
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable


CYRILLIC_RE = re.compile(r"[\u0400-\u052f]")
LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")
SPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
ALLOWED_PUNCTUATION = set(".,'’&+%°/:-()№")

INTERNATIONAL_LABEL_TERMS = [
    # English
    "Estate", "Estate Bottled", "Reserve", "Special Reserve", "Vintage",
    "Old Vines", "Single Vineyard", "Limited Edition", "Dry", "Semi Dry",
    "Semi Sweet", "Sweet", "Red Wine", "White Wine", "Rosé Wine",
    "Sparkling Wine", "Product of Italy", "Product of France",
    # French (keep accents: these exercise the new French-capable fonts)
    "Château", "Cuvée", "Réserve", "Grande Réserve", "Brut",
    "Extra Brut", "Demi-Sec", "Blanc de Blancs", "Blanc de Noirs",
    "Grand Cru", "Premier Cru", "Vieilles Vignes", "Vin de France",
    "Vin de Pays", "Mis en bouteille au château",
    "Appellation d’Origine Contrôlée",
    # Italian
    "Riserva", "Vino Rosso", "Vino Bianco", "Vino Rosato", "Spumante",
    "Prosecco", "Vendemmia", "Prodotto in Italia",
    "Denominazione di Origine Controllata", "Garantita",
    # Spanish / Portuguese
    "Reserva", "Gran Reserva", "Crianza", "Vino Tinto", "Vino Blanco",
    "Vino Rosado", "Denominación de Origen", "Vinho Verde", "Vinho Tinto",
    "Vinho Branco", "Quinta", "Colheita", "Garrafeira",
    # Varieties and regions missing or rare in the scraped catalogue
    "Cabernet Sauvignon", "Sauvignon Blanc", "Pinot Noir", "Pinot Grigio",
    "Chardonnay", "Merlot", "Syrah", "Shiraz", "Sangiovese", "Riesling",
    "Tempranillo", "Malbec", "Zinfandel", "Moscato", "Alvarinho",
    "Bordeaux", "Bourgogne", "Champagne", "Toscana", "Piemonte", "Veneto",
]

CYRILLIC_LABEL_TERMS = [
    "Вино", "Красное вино", "Белое вино", "Розовое вино", "Игристое вино",
    "Сухое", "Полусухое", "Полусладкое", "Сладкое", "Брют", "Экстра брют",
    "Выдержанное", "Коллекционное", "Резерв", "Специальный резерв",
    "Защищённое географическое указание", "Российское вино",
    "Вино защищённого наименования места происхождения",
]

NUMERIC_LABEL_TERMS = [
    "187 ml", "375 ml", "500 ml", "750 ml", "1 L", "1.5 L", "75 cl",
    "10% vol", "11% vol", "11.5% vol", "12% vol", "12.5% vol",
    "13% vol", "13.5% vol", "14% vol", "14.5% vol", "15% vol",
]


def clean_text(value: Any) -> str:
    """Normalize catalogue text without destroying accents or Cyrillic."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = CONTROL_RE.sub(" ", text).replace("\u00a0", " ")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = text.replace('“', '"').replace('”', '"').replace('`', "'")
    text = "".join(
        char
        if char.isalnum() or char.isspace() or char in ALLOWED_PUNCTUATION
        else " "
        for char in text
    )
    if text.count("(") != text.count(")"):
        text = text.replace("(", " ").replace(")", " ")
    return SPACE_RE.sub(" ", text).strip(" .,:;-/")


def script_of(text: str) -> str:
    has_cyrillic = bool(CYRILLIC_RE.search(text))
    has_latin = bool(LATIN_RE.search(text))
    if has_cyrillic and has_latin:
        return "mixed"
    if has_cyrillic:
        return "cyrillic"
    if has_latin:
        return "latin"
    if any(char.isdigit() for char in text):
        return "numeric"
    return "other"


def is_usable(text: str, min_length: int = 2) -> bool:
    if len(text) < min_length or URL_RE.search(text):
        return False
    if "\\" in text or text.count("/") > 2:
        return False
    return any(char.isalnum() for char in text)


def chunk_phrase(text: str, max_length: int) -> list[str]:
    """Split a long value at semantic punctuation, then at word boundaries."""
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= max_length:
        return [text]

    segments = [
        clean_text(part)
        for part in re.split(r"\s*[,;|]\s*|\s+[/-]\s+", text)
        if clean_text(part)
    ]
    chunks: list[str] = []
    for segment in segments:
        if len(segment) <= max_length:
            chunks.append(segment)
            continue

        current: list[str] = []
        for word in segment.split():
            if len(word) > max_length:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                continue
            candidate = " ".join([*current, word])
            if current and len(candidate) > max_length:
                chunks.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            chunks.append(" ".join(current))
    return [chunk for chunk in chunks if is_usable(chunk)]


def strip_known_suffix(name: str, suffixes: Iterable[str]) -> str:
    """Derive the prominent product-name line by removing category suffixes."""
    result = clean_text(name)
    suffix_set = {clean_text(value) for value in suffixes if value}
    for suffix in sorted(suffix_set, key=len, reverse=True):
        match = re.search(rf"(?:[,.-]?\s*){re.escape(suffix)}$", result, re.I)
        if match and match.start() > 0:
            result = result[: match.start()].strip(" ,.-")
            break
    return result


def nested_name(value: Any) -> str:
    if isinstance(value, dict):
        return clean_text(value.get("name"))
    return clean_text(value)


def load_wines(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for key in ("wines", "items", "results", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("Metadata JSON must be a list, or contain wines/items/results/data.")
    return [item for item in data if isinstance(item, dict)]


def build_lexicons(
    wines: list[dict[str, Any]], max_length: int, uppercase_variants: bool
) -> tuple[dict[str, set[str]], dict[str, Counter[str]]]:
    pools: dict[str, set[str]] = defaultdict(set)
    origins: dict[str, Counter[str]] = defaultdict(Counter)

    def add(
        value: Any,
        origin: str,
        *,
        split_long: bool = True,
        force_pool: str | None = None,
    ) -> None:
        text = clean_text(value)
        if not is_usable(text):
            return
        candidates = chunk_phrase(text, max_length) if split_long else [text]
        for candidate in candidates:
            if len(candidate) > max_length or not is_usable(candidate):
                continue
            script = force_pool or script_of(candidate)
            if script == "other":
                continue
            pools[script].add(candidate)
            origins[origin][script] += 1
            if uppercase_variants and script in {"latin", "cyrillic", "mixed"}:
                upper = candidate.upper()
                if upper != candidate and len(upper) <= max_length:
                    pools[script].add(upper)

    for wine in wines:
        detailed = wine.get("detailed_metadata") or {}
        raw = wine.get("raw_metadata") or {}

        sugar_values = [
            wine.get("sugar"), raw.get("category"),
            nested_name(detailed.get("category")),
        ]
        name = clean_text(wine.get("name") or raw.get("title") or detailed.get("title"))
        add(name, "name")
        derived_name = strip_known_suffix(name, sugar_values)
        if derived_name and derived_name.casefold() != name.casefold():
            add(derived_name, "derived_name")

        add(wine.get("producer"), "producer")
        add(raw.get("manufacturer"), "producer")
        add(nested_name(detailed.get("manufacturer")), "producer")

        add(wine.get("region"), "region")
        add(raw.get("region"), "region")
        add(nested_name(detailed.get("region")), "region")

        for value in sugar_values:
            add(value, "category")

        grape = wine.get("grape")
        if isinstance(grape, list):
            for value in grape:
                add(nested_name(value), "grape")
        else:
            add(nested_name(grape), "grape")
        for value in detailed.get("grapes") or []:
            add(nested_name(value), "grape")

        year = wine.get("year")
        if year:
            add(year, "year", split_long=False, force_pool="numeric")

        alcohol = detailed.get("alcohol")
        if isinstance(alcohol, (int, float)) and 0 < alcohol < 30:
            add(
                f"{alcohol:g}% vol", "alcohol",
                split_long=False, force_pool="numeric",
            )

    for term in INTERNATIONAL_LABEL_TERMS:
        add(term, "curated_latin")
    for term in CYRILLIC_LABEL_TERMS:
        add(term, "curated_cyrillic")
    for term in NUMERIC_LABEL_TERMS:
        add(term, "curated_numeric", split_long=False, force_pool="numeric")

    # Keep vintages separate so the generator can sample them at a modest rate.
    for year in range(1980, date.today().year + 2):
        add(str(year), "vintage", split_long=False, force_pool="numeric")

    return pools, origins


def sorted_lines(lines: Iterable[str]) -> list[str]:
    return sorted(set(lines), key=lambda value: (value.casefold(), value))


def write_lines(path: Path, lines: Iterable[str]) -> int:
    values = sorted_lines(lines)
    path.write_text("".join(f"{line}\n" for line in values), encoding="utf-8")
    return len(values)


def resolve_input(value: str) -> Path:
    requested = Path(value).expanduser()
    script_dir = Path(__file__).resolve().parent
    candidates = [requested, Path.cwd() / requested, script_dir / requested]
    if requested.name == "wines_metadata.json":
        candidates.extend([
            script_dir / "wines_metadata.json",
            script_dir / "dataset" / "wines_metadata.json",
        ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find metadata file: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset/wines_metadata.json")
    parser.add_argument("--output-dir", default="dataset/ocr_training/lexicons")
    parser.add_argument(
        "--max-length", type=int, default=32,
        help="Maximum characters per line (32 is suitable for PARSeq at 128x32).",
    )
    parser.add_argument(
        "--no-uppercase-variants", action="store_true",
        help="Do not add uppercase copies of alphabetic label lines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_length < 8:
        raise ValueError("--max-length must be at least 8")

    input_path = resolve_input(args.input)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Reading {input_path}")
    wines = load_wines(input_path)
    pools, origins = build_lexicons(
        wines, args.max_length, uppercase_variants=not args.no_uppercase_variants
    )

    files = {
        "latin": "wine_lexicon_latin.txt",
        "cyrillic": "wine_lexicon_cyrillic.txt",
        "mixed": "wine_lexicon_mixed.txt",
        "numeric": "wine_lexicon_numeric.txt",
    }
    counts: dict[str, int] = {}
    for script, filename in files.items():
        counts[script] = write_lines(output_dir / filename, pools.get(script, set()))

    alphabetic = pools.get("latin", set()) | pools.get("cyrillic", set()) | pools.get("mixed", set())
    counts["alphabetic"] = write_lines(output_dir / "wine_lexicon_alphabetic.txt", alphabetic)
    counts["all"] = write_lines(
        output_dir / "wine_lexicon_all.txt", alphabetic | pools.get("numeric", set())
    )

    manifest = {
        "source": str(input_path),
        "wine_records": len(wines),
        "max_line_length": args.max_length,
        "unicode_normalization": "NFC",
        "uppercase_variants": not args.no_uppercase_variants,
        "counts": counts,
        "files": {
            **files,
            "alphabetic": "wine_lexicon_alphabetic.txt",
            "all": "wine_lexicon_all.txt",
        },
        "recommended_sampling": {
            "latin": 0.40,
            "cyrillic": 0.40,
            "mixed": 0.10,
            "numeric": 0.10,
        },
        "notes": [
            "All files are deduplicated; balance by sampling pools, not by repeating lines.",
            "Use per-script pools with fonts that passed the matching coverage audit.",
            "Do not use the lexicon to silently overwrite raw OCR at inference time.",
        ],
        "candidate_counts_before_deduplication": {
            origin: dict(counter) for origin, counter in sorted(origins.items())
        },
    }
    (output_dir / "wine_lexicon_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[+] Processed {len(wines)} wine records")
    for key in ("latin", "cyrillic", "mixed", "numeric", "alphabetic", "all"):
        print(f"    {key:10s}: {counts[key]:5d}")
    print(f"[+] Saved lexicons to {output_dir}")


if __name__ == "__main__":
    main()

# python 2_lexicon_upgrade.py `
#   --input dataset/wines_metadata.json `
#   --output-dir dataset/ocr_training/lexicons