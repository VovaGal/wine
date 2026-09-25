"""
    python build_wine_catalog.py --input dataset/wines_metadata.json \
        --output dataset/wine_catalog.jsonl

To include verified labels not present in the scraped site, pass a JSON list
with ``--additions dataset/verified_wines.json``. Each addition needs a stable
``wine_id`` (prefixed ``manual:``) and ``name``; optional fields include
``producer``, ``region``, ``grapes``, ``year``, and verified full-name ``aliases``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from wine_normalizer import display_text, search_key


SCHEMA_VERSION = 1
EXACT_YEAR = re.compile(r"(?:19|20)\d{2}$")


def nested_name(value: Any) -> str:
    return display_text(value.get("name")) if isinstance(value, dict) else display_text(value)


def distinct_text(*values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = display_text(value)
        key = search_key(text)
        if key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def extract_year(value: Any) -> int | None:
    text = display_text(value)
    if not EXACT_YEAR.fullmatch(text):
        return None
    year = int(text)
    return year if 1900 <= year <= 2100 else None


def name_without_style(name: str, styles: list[str]) -> str:
    """Remove a known trailing category; do not invent standalone brand/grape aliases."""
    name_key = search_key(name)
    for style in sorted(styles, key=lambda item: len(search_key(item)), reverse=True):
        style_key = search_key(style)
        if style_key and name_key.endswith(" " + style_key):
            # Slice the original name to retain its spelling and accents.
            words = name.split()
            style_words = style.split()
            if len(words) > len(style_words):
                candidate = " ".join(words[:-len(style_words)]).rstrip(" ,.;-–—")
                if len(search_key(candidate)) >= 4:
                    return candidate
    return ""


def name_from_slug(slug: str) -> str:
    # Slugs may be transliterations useful for Latin OCR of Russian products.
    # Keep whole slug phrases, never individual words or numeric suffixes.
    text = display_text(slug).strip("/ ")
    if "/" in text or not re.fullmatch(r"[a-zA-Z0-9_-]+", text):
        return ""
    text = text.replace("_", " ").replace("-", " ")
    return text if len(search_key(text)) >= 6 and len(text.split()) >= 2 else ""


def wine_record(wine: dict[str, Any]) -> dict[str, Any]:
    raw = wine.get("raw_metadata") or {}
    detailed = wine.get("detailed_metadata") or {}
    if not isinstance(raw, dict) or not isinstance(detailed, dict):
        raise ValueError("raw_metadata and detailed_metadata must be objects")

    wine_id = display_text(wine.get("scraped_id"))
    name = display_text(wine.get("name") or raw.get("title") or detailed.get("title"))
    if not wine_id or not name:
        raise ValueError("Every wine needs scraped_id and name/title")

    styles = distinct_text(wine.get("sugar"), raw.get("category"), nested_name(detailed.get("category")))
    grapes_raw = detailed.get("grapes") or []
    if not isinstance(grapes_raw, list):
        grapes_raw = [grapes_raw]
    grapes_top = wine.get("grape") or []
    if not isinstance(grapes_top, list):
        grapes_top = [grapes_top]
    grapes = distinct_text(*(nested_name(g) for g in [*grapes_top, *grapes_raw]))

    producer = display_text(wine.get("producer") or raw.get("manufacturer") or nested_name(detailed.get("manufacturer")))
    region = display_text(wine.get("region") or raw.get("region") or nested_name(detailed.get("region")))
    slug = display_text(detailed.get("slug") or raw.get("slug"))

    # A scraped title might include a year even though the catalog year is null.
    # Extract it only from an explicit structured field, never from logo dates.
    year = extract_year(wine.get("year"))
    alcohol_raw = detailed.get("alcohol")
    alcohol = None
    if isinstance(alcohol_raw, (float, int)) and not isinstance(alcohol_raw, bool):
        if 0 < alcohol_raw < 30:
            alcohol = float(alcohol_raw)

    alias_sources = [
        (name, "name"),
        (display_text(raw.get("title")), "raw_title"),
        (display_text(detailed.get("title")), "detailed_title"),
        (name_without_style(name, styles), "name_without_style"),
        (name_from_slug(slug), "slug"),
    ]
    aliases: list[dict[str, str]] = []
    seen_aliases: set[str] = set()
    for text, source in alias_sources:
        key = search_key(text)
        minimum = 2 if source in {"name", "raw_title", "detailed_title"} else 4
        if len(key) < minimum or key in seen_aliases:
            continue
        seen_aliases.add(key)
        aliases.append({"text": text, "key": key, "source": source})

    # These descriptors intentionally stay outside aliases: e.g. 'Riserva'
    # or '750 ml' alone is shared by many different bottles.
    vector_parts = distinct_text(name, producer, *grapes, region, *styles)
    return {
        "schema_version": SCHEMA_VERSION,
        "wine_id": wine_id,
        "name": name,
        "producer": producer or None,
        "region": region or None,
        "style": styles,
        "color": display_text(wine.get("color")) or None,
        "grapes": grapes,
        "year": year,
        "alcohol_percent": alcohol,
        "slug": slug or None,
        "image_url": display_text(wine.get("remote_image_url")) or None,
        "aliases": aliases,
        "search_text": " | ".join(vector_parts),
    }


def build_catalog(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected a top-level JSON list of wines")
    products: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, wine in enumerate(data):
        if not isinstance(wine, dict):
            raise ValueError(f"Wine at position {index} is not an object")
        try:
            product = wine_record(wine)
        except ValueError as exc:
            raise ValueError(f"Wine at position {index}: {exc}") from exc
        if product["wine_id"] in seen_ids:
            raise ValueError(f"Duplicate scraped_id: {product['wine_id']}")
        seen_ids.add(product["wine_id"])
        products.append(product)
    return products


def load_verified_additions(path: Path, existing_ids: set[str]) -> list[dict[str, Any]]:
    """Normalize explicitly verified bottles into the same catalog schema."""
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("--additions must be a JSON list")
    products = []
    for position, item in enumerate(data, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Verified addition {position} must be an object")
        wine_id = display_text(item.get("wine_id"))
        if not wine_id.startswith("manual:") or len(wine_id) <= len("manual:"):
            raise ValueError(f"Verified addition {position} needs a manual: wine_id")
        if wine_id in existing_ids:
            raise ValueError(f"Duplicate wine_id: {wine_id}")
        if not display_text(item.get("name")):
            raise ValueError(f"Verified addition {position} needs a name")
        grapes = item.get("grapes", [])
        aliases = item.get("aliases", [])
        if not isinstance(grapes, list) or not all(isinstance(g, str) for g in grapes):
            raise ValueError(f"Verified addition {position}: grapes must be a list of strings")
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise ValueError(f"Verified addition {position}: aliases must be a list of strings")
        product = wine_record({
            "scraped_id": wine_id,
            "name": item["name"],
            "producer": item.get("producer"),
            "region": item.get("region"),
            "grape": grapes,
            "year": item.get("year"),
            "sugar": item.get("style"),
            "remote_image_url": item.get("image_url"),
        })
        seen = {alias["key"] for alias in product["aliases"]}
        for alias in aliases:
            key = search_key(alias)
            if len(key) < 6 or len(key.split()) < 2:
                raise ValueError(
                    f"Verified addition {position}: aliases must be full, "
                    "specific product-name phrases"
                )
            if key not in seen:
                product["aliases"].append({"text": display_text(alias), "key": key, "source": "verified"})
                seen.add(key)
        product["source"] = "verified_addition"
        existing_ids.add(wine_id)
        products.append(product)
    return products


def write_catalog(products: list[dict[str, Any]], output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    # Atomic replacement: a crash cannot leave a partly written catalog.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for product in products:
                handle.write(json.dumps(product, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/wines_metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("dataset/wine_catalog.jsonl"))
    parser.add_argument("--additions", type=Path, help="Verified bottles absent from the scraped metadata (JSON list)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing catalog")
    args = parser.parse_args()

    products = build_catalog(args.input)
    if args.additions is not None:
        products.extend(load_verified_additions(args.additions, {p["wine_id"] for p in products}))
    write_catalog(products, args.output, args.overwrite)
    alias_count = sum(len(p["aliases"]) for p in products)
    print(f"[+] Wrote {len(products):,} wines and {alias_count:,} name aliases to {args.output}")
    print("[*] Details: " + ", ".join(
        f"{key}={value:,}" for key, value in sorted(Counter(
            alias["source"] for product in products for alias in product["aliases"]
        ).items())
    ))
    print("[*] Unknown vintages remain null; shared descriptors are not product aliases.")


if __name__ == "__main__":
    main()
