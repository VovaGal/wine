"""Download and audit an OCR font collection for wine-label synthesis.

The script downloads open-source fonts from the official Google Fonts GitHub
repository, checks their real Unicode cmap, and keeps the main directory flat
for compatibility with the existing TRDG command. Audited pools are created in
a sibling directory:

    dataset/ocr_training/fonts/                 # all fonts; existing TRDG path
    dataset/ocr_training/font_pools/latin/
    dataset/ocr_training/font_pools/french/
    dataset/ocr_training/font_pools/cyrillic/
    dataset/ocr_training/font_pools/multilingual/

Install the only extra dependency with:
    pip install fonttools

Examples:
    python fonts_loader.py
    python fonts_loader.py --output dataset/ocr_training/fonts
    python fonts_loader.py --audit-only --output dataset/ocr_training/fonts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from fontTools.ttLib import TTFont
except ImportError as exc:
    raise SystemExit(
        "fonttools is required. Install it with: pip install fonttools"
    ) from exc


GOOGLE_FONTS_API = "https://api.github.com/repos/google/fonts/contents"
REPOSITORY_ROOTS = ("ofl", "apache", "ufl")
FONT_EXTENSIONS = {".ttf", ".otf"}

# Structural variety matters more than downloading hundreds of near-identical
# fonts. Existing fonts can remain in the destination; the audit includes them.
FONT_FAMILIES = (
    # High-contrast / luxury serif
    ("Bodoni Moda", "bodonimoda", "didone"),
    ("Libre Bodoni", "librebodoni", "didone"),
    ("Playfair Display", "playfairdisplay", "didone"),
    ("DM Serif Display", "dmserifdisplay", "display-serif"),
    ("Cormorant Garamond", "cormorantgaramond", "display-serif"),
    ("Prata", "prata", "display-serif"),
    ("Yeseva One", "yesevaone", "display-serif"),
    ("Oranienbaum", "oranienbaum", "display-serif"),
    # Traditional serif / body copy
    ("EB Garamond", "ebgaramond", "serif"),
    ("Libre Baskerville", "librebaskerville", "serif"),
    ("Lora", "lora", "serif"),
    ("Old Standard TT", "oldstandardtt", "serif"),
    ("PT Serif", "ptserif", "serif"),
    ("Philosopher", "philosopher", "serif"),
    ("Forum", "forum", "serif"),
    # Slab serif
    ("Roboto Slab", "robotoslab", "slab-serif"),
    ("Bitter", "bitter", "slab-serif"),
    ("Arvo", "arvo", "slab-serif"),
    ("Kelly Slab", "kellyslab", "slab-serif"),
    ("Kurale", "kurale", "slab-serif"),
    # Condensed / geometric / widely tracked capitals
    ("Oswald", "oswald", "condensed-sans"),
    ("Roboto Condensed", "robotocondensed", "condensed-sans"),
    ("PT Sans Narrow", "ptsansnarrow", "condensed-sans"),
    ("Bebas Neue", "bebasneue", "condensed-display"),
    ("Montserrat", "montserrat", "geometric-sans"),
    ("Raleway", "raleway", "geometric-sans"),
    ("Poppins", "poppins", "geometric-sans"),
    ("Comfortaa", "comfortaa", "geometric-sans"),
    ("Poiret One", "poiretone", "thin-display"),
    ("Russo One", "russoone", "display-sans"),
    # Classical/decorative capitals
    ("Cinzel", "cinzel", "classical-display"),
    ("Cinzel Decorative", "cinzeldecorative", "classical-display"),
    # Formal script / connected lettering
    ("Great Vibes", "greatvibes", "formal-script"),
    ("Allura", "allura", "formal-script"),
    ("Dancing Script", "dancingscript", "script"),
    ("Lobster", "lobster", "script"),
    ("Pacifico", "pacifico", "brush-script"),
    ("Pattaya", "pattaya", "brush-script"),
    # Handwritten / rough display
    ("Caveat", "caveat", "handwritten"),
    ("Bad Script", "badscript", "handwritten"),
    ("Marck Script", "marckscript", "handwritten"),
    ("Neucha", "neucha", "handwritten"),
    ("Amatic SC", "amaticsc", "handwritten-display"),
)

ENGLISH_REQUIRED = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)
FRENCH_REQUIRED = ENGLISH_REQUIRED | set(
    "ÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸàâæçéèêëîïôœùûüÿ"
)
CYRILLIC_REQUIRED = set(
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "0123456789"
)


@dataclass
class FontReport:
    filename: str
    family_group: str
    sha256: str
    english_coverage: float
    french_coverage: float
    cyrillic_coverage: float
    supports_english: bool
    supports_french: bool
    supports_cyrillic: bool
    missing_french: str
    missing_cyrillic: str
    error: str = ""


def request_json(url: str, retries: int = 3):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "wine-ocr-font-builder/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code in {403, 429} and attempt + 1 < retries:
                time.sleep(2 ** attempt)
                continue
            raise
        except (TimeoutError, urllib.error.URLError):
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    return None


def download_file(url: str, destination: Path):
    request = urllib.request.Request(
        url, headers={"User-Agent": "wine-ocr-font-builder/1.0"}
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=60) as response:
        temporary.write_bytes(response.read())
    temporary.replace(destination)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def find_family_files(slug: str):
    for root in REPOSITORY_ROOTS:
        url = f"{GOOGLE_FONTS_API}/{root}/{slug}"
        entries = request_json(url)
        if entries:
            font_entries = [
                entry
                for entry in entries
                if entry.get("type") == "file"
                and Path(entry.get("name", "")).suffix.casefold() in FONT_EXTENSIONS
            ]
            if font_entries:
                return root, font_entries

            # A few families keep static faces in a nested directory.
            static_entry = next(
                (
                    entry
                    for entry in entries
                    if entry.get("type") == "dir" and entry.get("name") == "static"
                ),
                None,
            )
            if static_entry:
                static_files = request_json(static_entry["url"]) or []
                font_entries = [
                    entry
                    for entry in static_files
                    if entry.get("type") == "file"
                    and Path(entry.get("name", "")).suffix.casefold()
                    in FONT_EXTENSIONS
                ]
                if font_entries:
                    return root, font_entries
    return None, []


def download_families(all_dir: Path):
    all_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    missing_families = []

    for family_name, slug, category in FONT_FAMILIES:
        print(f"[*] {family_name} ({category})")
        try:
            root, entries = find_family_files(slug)
        except Exception as exc:
            print(f"    [!] Metadata request failed: {exc}")
            missing_families.append(family_name)
            continue

        if not entries:
            print("    [!] Family not found in the expected repository roots.")
            missing_families.append(family_name)
            continue

        for entry in entries:
            source_name = entry["name"]
            destination_name = (
                f"{safe_name(slug)}__{safe_name(category)}__{safe_name(source_name)}"
            )
            destination = all_dir / destination_name
            if destination.exists() and destination.stat().st_size > 0:
                print(f"    [=] {destination.name}")
                continue
            try:
                download_file(entry["download_url"], destination)
                downloaded += 1
                print(f"    [+] {destination.name} [{root}]")
            except Exception as exc:
                print(f"    [!] Download failed for {source_name}: {exc}")

    return downloaded, missing_families


def coverage(codepoints: set[int], required: set[str]):
    present = {chr(codepoint) for codepoint in codepoints}
    missing = sorted(required - present)
    ratio = 1.0 - len(missing) / len(required)
    return ratio, missing


def inspect_font(font_path: Path) -> FontReport:
    category_parts = font_path.stem.split("__", 2)
    category = category_parts[1] if len(category_parts) >= 3 else "existing"
    digest = hashlib.sha256(font_path.read_bytes()).hexdigest()

    try:
        font = TTFont(font_path, lazy=True)
        codepoints = set()
        for table in font["cmap"].tables:
            if table.isUnicode():
                codepoints.update(table.cmap)
        font.close()

        english_ratio, _ = coverage(codepoints, ENGLISH_REQUIRED)
        french_ratio, missing_french = coverage(codepoints, FRENCH_REQUIRED)
        cyrillic_ratio, missing_cyrillic = coverage(codepoints, CYRILLIC_REQUIRED)
        return FontReport(
            filename=font_path.name,
            family_group=category,
            sha256=digest,
            english_coverage=english_ratio,
            french_coverage=french_ratio,
            cyrillic_coverage=cyrillic_ratio,
            supports_english=english_ratio == 1.0,
            supports_french=french_ratio == 1.0,
            supports_cyrillic=cyrillic_ratio == 1.0,
            missing_french="".join(missing_french),
            missing_cyrillic="".join(missing_cyrillic),
        )
    except Exception as exc:
        return FontReport(
            filename=font_path.name,
            family_group=category,
            sha256=digest,
            english_coverage=0.0,
            french_coverage=0.0,
            cyrillic_coverage=0.0,
            supports_english=False,
            supports_french=False,
            supports_cyrillic=False,
            missing_french="",
            missing_cyrillic="",
            error=str(exc),
        )


def copy_to_pool(source: Path, pool: Path):
    pool.mkdir(parents=True, exist_ok=True)
    destination = pool / source.name
    if not destination.exists() or source.stat().st_size != destination.stat().st_size:
        shutil.copy2(source, destination)


def audit_and_build_pools(output_root: Path):
    all_dir = output_root
    pools_root = output_root.parent / "font_pools"
    pool_dirs = {
        "latin": pools_root / "latin",
        "french": pools_root / "french",
        "cyrillic": pools_root / "cyrillic",
        "multilingual": pools_root / "multilingual",
    }
    for pool in pool_dirs.values():
        pool.mkdir(parents=True, exist_ok=True)

    font_paths = sorted(
        path
        for path in all_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in FONT_EXTENSIONS
    )
    if not font_paths:
        raise RuntimeError(f"No TTF/OTF files found in {all_dir}")

    reports = []
    seen_hashes = set()
    duplicates = 0
    for font_path in font_paths:
        report = inspect_font(font_path)
        if report.sha256 in seen_hashes:
            duplicates += 1
            print(f"[=] Duplicate bytes skipped: {font_path.name}")
            continue
        seen_hashes.add(report.sha256)
        reports.append(report)

        if report.error:
            print(f"[!] Invalid font {font_path.name}: {report.error}")
            continue
        if report.supports_english:
            copy_to_pool(font_path, pool_dirs["latin"])
        if report.supports_french:
            copy_to_pool(font_path, pool_dirs["french"])
        if report.supports_cyrillic:
            copy_to_pool(font_path, pool_dirs["cyrillic"])
        if report.supports_french and report.supports_cyrillic:
            copy_to_pool(font_path, pool_dirs["multilingual"])

    csv_path = output_root / "font_coverage.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=asdict(reports[0]).keys())
        writer.writeheader()
        for report in reports:
            row = asdict(report)
            row["english_coverage"] = f"{report.english_coverage:.4f}"
            row["french_coverage"] = f"{report.french_coverage:.4f}"
            row["cyrillic_coverage"] = f"{report.cyrillic_coverage:.4f}"
            writer.writerow(row)

    manifest_path = output_root / "font_manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(report) for report in reports], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    counts = {
        name: len(list(path.glob("*.ttf"))) + len(list(path.glob("*.otf")))
        for name, path in pool_dirs.items()
    }
    print("\n" + "=" * 64)
    print(f"Audited unique fonts: {len(reports)} (duplicate files: {duplicates})")
    for name, count in counts.items():
        print(f"{name:>12}: {count}")
    print(f"Coverage report: {csv_path}")
    print(f"JSON manifest:   {manifest_path}")
    print("=" * 64)
    return reports


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and audit Google Fonts for wine-label OCR training."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/ocr_training/fonts"),
        help="Flat TRDG font directory; audited pools are created beside it.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Skip downloads and rebuild reports/pools from the output folder.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = args.output.resolve()
    all_dir = output_root
    all_dir.mkdir(parents=True, exist_ok=True)

    if not args.audit_only:
        downloaded, missing = download_families(all_dir)
        print(f"[*] Downloaded {downloaded} new font files.")
        if missing:
            print("[!] Families not downloaded: " + ", ".join(missing))

    audit_and_build_pools(output_root)


if __name__ == "__main__":
    main()
