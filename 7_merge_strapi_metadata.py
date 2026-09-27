"""
    python 7_merge_strapi_metadata.py --base dataset/wines_metadata.json \
        --strapi strapi_output0709.csv --output dataset/wines_metadata_merged.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path

REQUIRED = (
    "Название вина", "Категория", "Цвет", "Регион", "Сорт винограда",
    "Описание", "Винодельня", "Slug", "Название фото",
)


def read_json_list(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"Expected a list of metadata objects in {path}")
    return data


def slug_of(row: dict) -> str:
    return str((row.get("raw_metadata") or {}).get("slug") or
               (row.get("detailed_metadata") or {}).get("slug") or "").strip()


def new_record(csv_row: dict[str, str], scraped_id: str) -> dict:
    name = csv_row["Название вина"]
    winery = csv_row["Винодельня"]
    region = csv_row["Регион"]
    category = csv_row["Категория"]
    grapes = [piece.strip() for piece in csv_row["Сорт винограда"].split(",") if piece.strip()]
    slug = csv_row["Slug"]
    return {
        "scraped_id": scraped_id,
        "name": name,
        "producer": winery,
        "year": None,  # The CSV has no separate vintage field.
        "color": csv_row["Цвет"],
        "sugar": None,  # 'Категория' is a colour group, not a sweetness level.
        "region": region,
        "grape": grapes,
        "rating_roskachestvo": None,
        "remote_image_url": None,  # A photo filename is not a downloadable URL.
        "local_image_path": None,
        "raw_metadata": {
            "title": name, "manufacturer": winery, "region": region,
            "category": category, "color": csv_row["Цвет"], "slug": slug,
            "photo_filename": csv_row["Название фото"],
        },
        "detailed_metadata": {
            "title": name, "manufacturer": {"name": winery},
            "region": {"name": region}, "category": {"name": category},
            "grapes": [{"name": grape} for grape in grapes],
            "description": csv_row["Описание"], "slug": slug,
        },
        "metadata_source": "strapi_output0709.csv",
    }


def merge(base_path: Path, csv_path: Path, output_path: Path) -> tuple[int, int, int]:
    base = read_json_list(base_path)
    base_by_slug = {slug_of(row): row for row in base}
    if "" in base_by_slug or len(base_by_slug) != len(base):
        raise ValueError("Base metadata has missing or duplicate slugs")
    ids = [int(str(row.get("scraped_id"))) for row in base]
    if len(set(ids)) != len(ids):
        raise ValueError("Base metadata has duplicate scraped_id values")

    # Preserve identities when regenerating after a later CSV update.
    previously_assigned: dict[str, str] = {}
    if output_path.is_file():
        for row in read_json_list(output_path):
            slug = slug_of(row)
            if slug and slug not in base_by_slug:
                previously_assigned[slug] = str(row["scraped_id"])
                ids.append(int(previously_assigned[slug]))
    next_id = max(ids, default=0) + 1

    csv_by_slug: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(REQUIRED).issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(REQUIRED)}")
        for row in reader:
            clean = {key: (value or "").strip() for key, value in row.items()}
            slug = clean["Slug"]
            if not slug or not re.fullmatch(r"[A-Za-z0-9_-]+", slug):
                raise ValueError(f"Missing or unsafe slug: {slug!r}")
            if slug in csv_by_slug and csv_by_slug[slug] != clean:
                raise ValueError(f"Conflicting CSV records for slug {slug}")
            csv_by_slug[slug] = clean

    for slug in base_by_slug.keys() & csv_by_slug.keys():
        original = base_by_slug[slug]
        supplied = csv_by_slug[slug]
        if (str(original.get("name") or "").strip() != supplied["Название вина"] or
                str(original.get("producer") or "").strip() != supplied["Винодельня"]):
            raise ValueError(f"Existing slug {slug} has a changed name or winery; review manually")

    additions = []
    for slug in sorted(csv_by_slug.keys() - base_by_slug.keys()):
        assigned = previously_assigned.get(slug)
        if assigned is None:
            assigned = str(next_id)
            next_id += 1
        additions.append(new_record(csv_by_slug[slug], assigned))

    merged = [*base, *additions]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".wine_metadata_", suffix=".json", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(merged, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, output_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return len(base), len(additions), len(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--strapi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.base.resolve(), args.strapi.resolve()}:
        parser.error("Output must be a separate file; never overwrite the source")
    base_count, added, total = merge(args.base, args.strapi, args.output)
    print(f"Preserved {base_count} existing wines, added {added}, total {total}: {args.output}")


if __name__ == "__main__":
    main()
