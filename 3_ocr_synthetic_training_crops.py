"""Generate balanced wine-label OCR crops from the split lexicon pools.

This keeps the filename convention expected by the current PARSeq trainer:
    Exact label text_pool-0000123.jpg

It also writes labels.tsv. The next training revision should use that manifest
for grouped train/validation splitting instead of parsing filenames.

Example:
    python ocr_synthetic_training_crops.py --count 50000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import PIL
import PIL.ImageFont
from PIL import Image, ImageFilter

try:
    import cv2
except ImportError:  # The basic pipeline still works without geometric warps.
    cv2 = None


# Older TRDG releases still call FreeTypeFont.getsize(), removed by Pillow 10.
if not hasattr(PIL.ImageFont.FreeTypeFont, "getsize"):
    def getsize(self, text, *args, **kwargs):
        bbox = self.getbbox(text)
        if bbox is None:
            return (0, 0)
        return round(self.getlength(text)), bbox[3]

    PIL.ImageFont.FreeTypeFont.getsize = getsize


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FONT_SUFFIXES = {".ttf", ".otf"}
WINDOWS_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
SPACE_RE = re.compile(r"\s+")

LIGHT_BACKGROUNDS = [
    (250, 247, 239), (244, 239, 225), (238, 231, 218),
    (235, 224, 222), (226, 226, 221), (247, 241, 230),
]
DARK_BACKGROUNDS = [
    (25, 20, 20), (39, 18, 23), (21, 27, 33),
    (46, 38, 31), (31, 31, 30), (52, 23, 28),
]
DARK_INKS = [
    (22, 20, 18), (73, 18, 29), (31, 42, 54),
    (91, 52, 31), (82, 67, 25), (47, 29, 33),
]
LIGHT_INKS = [
    (250, 247, 239), (235, 223, 199), (232, 218, 177),
    (221, 221, 216), (241, 224, 226),
]
METALLIC_INKS = [
    (181, 139, 43), (204, 166, 72), (163, 151, 119),
    (201, 196, 178), (151, 92, 45),
]


@dataclass(frozen=True)
class Job:
    name: str
    pool: str
    count: int
    lexicon: Path
    fonts: Path
    language: str


def project_path(value: str) -> Path:
    """Resolve paths from the launch directory, matching the old script."""
    return Path(value).expanduser().resolve()


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Lexicon not found: {path}")
    seen: set[str] = set()
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = SPACE_RE.sub(" ", raw).strip()
        # TRDG uses label text in filenames. Normalize Windows-forbidden
        # punctuation now so generation cannot fail halfway through.
        line = SPACE_RE.sub(" ", WINDOWS_UNSAFE.sub(" ", line)).strip(" .")
        if 1 < len(line) <= 32 and line not in seen:
            seen.add(line)
            lines.append(line)
    if not lines:
        raise RuntimeError(f"Lexicon contains no usable lines: {path}")
    return lines


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def contains_extended_latin(text: str) -> bool:
    return any(ord(char) > 127 and char.isalpha() for char in text)


def has_fonts(path: Path) -> bool:
    return path.is_dir() and any(
        item.is_file() and item.suffix.casefold() in FONT_SUFFIXES
        for item in path.iterdir()
    )


def choose_font_pool(font_root: Path, preferred: str, flat_fonts: Path) -> Path:
    candidates = [font_root / preferred]
    if preferred in {"french", "cyrillic"}:
        candidates.append(font_root / "multilingual")
    candidates.extend([font_root / "latin", flat_fonts])
    for candidate in candidates:
        if has_fonts(candidate):
            return candidate
    raise FileNotFoundError(
        f"No fonts found for pool '{preferred}'. Run fonts_loader.py first."
    )


def allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    result = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(raw, key=lambda name: raw[name] - result[name], reverse=True)
    for name in order[:remainder]:
        result[name] += 1
    return result


def build_jobs(
    count: int,
    lexicon_dir: Path,
    font_root: Path,
    flat_fonts: Path,
    work_dir: Path,
) -> list[Job]:
    ratios = {"latin": 0.40, "cyrillic": 0.40, "mixed": 0.10, "numeric": 0.10}
    counts = allocate_counts(count, ratios)

    latin = read_lines(lexicon_dir / "wine_lexicon_latin.txt")
    latin_extended = [line for line in latin if contains_extended_latin(line)]
    latin_ascii = [line for line in latin if not contains_extended_latin(line)]
    cyrillic = read_lines(lexicon_dir / "wine_lexicon_cyrillic.txt")
    mixed = read_lines(lexicon_dir / "wine_lexicon_mixed.txt")
    numeric = read_lines(lexicon_dir / "wine_lexicon_numeric.txt")

    derived = work_dir / "derived_lexicons"
    derived.mkdir(parents=True, exist_ok=True)
    sources = {
        "latin_ascii": latin_ascii,
        "latin_extended": latin_extended,
        "cyrillic": cyrillic,
        "mixed": mixed,
        "numeric": numeric,
    }
    for name, lines in sources.items():
        if lines:
            write_lines(derived / f"{name}.txt", lines)

    # Guarantee meaningful accented-French exposure instead of leaving it to
    # the relative number of dictionary entries.
    extended_count = 0
    if latin_extended:
        extended_count = max(1, round(counts["latin"] * 0.25))
    ascii_count = counts["latin"] - extended_count

    specs = [
        ("latin_ascii", "latin", ascii_count, "latin_ascii", "latin", "en"),
        ("latin_extended", "latin", extended_count, "latin_extended", "french", "fr"),
        ("cyrillic", "cyrillic", counts["cyrillic"], "cyrillic", "cyrillic", "ru"),
        ("mixed", "mixed", counts["mixed"], "mixed", "multilingual", "ru"),
        ("numeric", "numeric", counts["numeric"], "numeric", "latin", "en"),
    ]
    jobs: list[Job] = []
    for name, pool, job_count, lexicon_name, font_name, language in specs:
        if job_count <= 0:
            continue
        lexicon_path = derived / f"{lexicon_name}.txt"
        if not lexicon_path.is_file():
            continue
        jobs.append(Job(
            name=name,
            pool=pool,
            count=job_count,
            lexicon=lexicon_path,
            fonts=choose_font_pool(font_root, font_name, flat_fonts),
            language=language,
        ))
    if sum(job.count for job in jobs) != count:
        raise RuntimeError("Internal error: generation job counts do not sum to --count")
    return jobs


def run_trdg(job: Job, output_dir: Path, workers: int) -> None:
    """Run TRDG in-process so the Pillow compatibility patch is inherited."""
    try:
        from trdg.run import main as trdg_main
    except ImportError as exc:
        raise RuntimeError(
            "TRDG is not installed. Activate the project venv and run "
            "'pip install trdg'."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    previous_argv = sys.argv[:]
    try:
        sys.argv = [
            "trdg",
            "-c", str(job.count),
            "-l", job.language,
            "-f", "32",
            "-w", "1",
            "-t", str(workers),
            "-fd", str(job.fonts),
            "-dt", str(job.lexicon),
            "--output_dir", str(output_dir),
            "-k", "2",
            "-rk",
            "-bl", "1",
            "-rbl",
            "-tc", "#171512,#42111D,#1F2A36,#5B341F,#6A571D",
            "-b", "1",
            "-d", "1",
        ]
        trdg_main()
    finally:
        sys.argv = previous_argv


def deterministic_seed(global_seed: int, job_name: str, source_name: str) -> int:
    payload = f"{global_seed}|{job_name}|{source_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def recolor_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    """Recolor TRDG's dark-on-white render as plausible label stock and ink."""
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    ink_alpha = np.clip((0.94 - gray) / 0.82, 0.0, 1.0)[..., None]

    dark_label = rng.random() < 0.25
    metallic = rng.random() < 0.18
    if dark_label:
        background = np.array(rng.choice(DARK_BACKGROUNDS), dtype=np.float32)
        ink = np.array(
            rng.choice(METALLIC_INKS if metallic else LIGHT_INKS), dtype=np.float32
        )
    else:
        background = np.array(rng.choice(LIGHT_BACKGROUNDS), dtype=np.float32)
        ink = np.array(
            rng.choice(METALLIC_INKS if metallic else DARK_INKS), dtype=np.float32
        )

    array = background * (1.0 - ink_alpha) + ink * ink_alpha
    height, width = gray.shape

    # Fine paper/print texture and a broad reflection band mimic curved glass
    # without making every training example artificially difficult.
    noise = np.random.default_rng(rng.randrange(2**32)).normal(
        0.0, rng.uniform(0.8, 3.2), size=(height, width, 1)
    )
    array += noise
    if rng.random() < 0.35:
        center = rng.uniform(-0.15 * width, 1.15 * width)
        sigma = rng.uniform(0.10 * width, 0.30 * width)
        strength = rng.uniform(3.0, 15.0) * (-1 if dark_label else 1)
        x = np.arange(width, dtype=np.float32)
        glare = np.exp(-0.5 * ((x - center) / max(sigma, 1.0)) ** 2)
        array += glare[None, :, None] * strength

    return Image.fromarray(np.uint8(np.clip(array, 0, 255)), mode="RGB")


def geometric_warp(image: Image.Image, rng: random.Random) -> Image.Image:
    if cv2 is None or rng.random() >= 0.55:
        return image
    array = np.asarray(image)
    height, width = array.shape[:2]
    if width < 12 or height < 8:
        return image

    # Very small perspective changes; large warps teach the recognizer noise.
    jitter_x = max(1.0, width * rng.uniform(0.005, 0.025))
    jitter_y = max(0.5, height * rng.uniform(0.01, 0.06))
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    target = np.float32([
        [rng.uniform(0, jitter_x), rng.uniform(0, jitter_y)],
        [width - 1 - rng.uniform(0, jitter_x), rng.uniform(0, jitter_y)],
        [width - 1 - rng.uniform(0, jitter_x), height - 1 - rng.uniform(0, jitter_y)],
        [rng.uniform(0, jitter_x), height - 1 - rng.uniform(0, jitter_y)],
    ])
    matrix = cv2.getPerspectiveTransform(source, target)
    border = tuple(int(value) for value in np.median(array.reshape(-1, 3), axis=0))
    warped = cv2.warpPerspective(
        array, matrix, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    )

    if rng.random() < 0.45:
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        amplitude = rng.uniform(-1.2, 1.2)
        curve = amplitude * np.cos((grid_x / max(width - 1, 1) - 0.5) * math.pi)
        warped = cv2.remap(
            warped, grid_x, grid_y - curve.astype(np.float32),
            interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101,
        )
    return Image.fromarray(warped, mode="RGB")


def save_with_degradation(image: Image.Image, destination: Path, rng: random.Random) -> None:
    if rng.random() < 0.50:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.05, 0.55)))
    quality = rng.randint(68, 96)
    # Encode once in memory to make JPEG degradation explicit and controlled.
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=rng.choice([0, 1, 2]))
    destination.write_bytes(buffer.getvalue())


def process_one(task: tuple[Path, Path, str, str, int, int]) -> tuple[str, str, str]:
    source, output_dir, label, pool, index, seed = task
    rng = random.Random(deterministic_seed(seed, pool, source.name))
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    image = recolor_crop(image, rng)
    image = geometric_warp(image, rng)
    filename = f"{label}_{pool}-{index:07d}.jpg"
    save_with_degradation(image, output_dir / filename, rng)
    return filename, label, pool


def collect_staged_images(path: Path) -> list[Path]:
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES
    )


def label_from_trdg_filename(path: Path) -> str:
    if "_" not in path.stem:
        raise ValueError(f"Unexpected TRDG filename: {path.name}")
    return path.stem.rsplit("_", 1)[0]


def prepare_output(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    generated = [
        item for item in path.iterdir()
        if item.is_file()
        and (item.suffix.casefold() in IMAGE_SUFFIXES or item.name in {"labels.tsv", "generation_manifest.json"})
    ]
    if generated and not overwrite:
        raise FileExistsError(
            f"{path} already contains generated data. Choose another --output-dir "
            "or pass --overwrite explicitly."
        )
    if overwrite:
        for item in generated:
            item.unlink()


def write_manifest(
    output_dir: Path,
    rows: list[tuple[str, str, str]],
    jobs: list[Job],
    args: argparse.Namespace,
) -> None:
    with (output_dir / "labels.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["filename", "label", "pool"])
        writer.writerows(rows)

    actual = {name: 0 for name in {job.pool for job in jobs}}
    for _, _, pool in rows:
        actual[pool] = actual.get(pool, 0) + 1
    manifest = {
        "requested_count": args.count,
        "generated_count": len(rows),
        "seed": args.seed,
        "image_height": 32,
        "opencv_warps_enabled": cv2 is not None,
        "pool_counts": actual,
        "jobs": [
            {
                **asdict(job),
                "lexicon": str(job.lexicon),
                "fonts": str(job.fonts),
            }
            for job in jobs
        ],
        "sampling_target": {
            "latin": 0.40,
            "cyrillic": 0.40,
            "mixed": 0.10,
            "numeric": 0.10,
            "latin_extended_share_within_latin": 0.25,
        },
    }
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50_000)
    parser.add_argument("--lexicon-dir", default="dataset/ocr_training/lexicons")
    parser.add_argument("--font-pools", default="dataset/ocr_training/font_pools")
    parser.add_argument("--flat-fonts", default="dataset/ocr_training/fonts")
    parser.add_argument("--output-dir", default="dataset/ocr_training/dated_synth_crops")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete previously generated images/manifests in --output-dir first.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate paths and print the job plan without invoking TRDG.",
    )
    parser.add_argument(
        "--skip-postprocess", action="store_true",
        help="Copy raw TRDG images without wine-label recoloring/warping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 100:
        raise ValueError("--count must be at least 100")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    lexicon_dir = project_path(args.lexicon_dir)
    font_root = project_path(args.font_pools)
    flat_fonts = project_path(args.flat_fonts)
    output_dir = project_path(args.output_dir)

    print(f"[*] Pillow: {PIL.__version__}")
    print(f"[*] OpenCV warps: {'enabled' if cv2 is not None else 'disabled'}")
    print(f"[*] Requested samples: {args.count:,}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wine_ocr_gen_", dir=output_dir.parent) as temporary:
        work_dir = Path(temporary)
        jobs = build_jobs(args.count, lexicon_dir, font_root, flat_fonts, work_dir)
        print("[*] Generation plan:")
        for job in jobs:
            print(
                f"    {job.name:15s} {job.count:6d}  "
                f"fonts={job.fonts.name}"
            )
        if args.dry_run:
            print("[+] Dry run passed; no images were generated.")
            return

        prepare_output(output_dir, args.overwrite)
        all_rows: list[tuple[str, str, str]] = []
        final_index = 0

        for job in jobs:
            staging = work_dir / "staging" / job.name
            print(f"[*] TRDG: {job.name} ({job.count:,} samples)")
            run_trdg(job, staging, args.workers)
            sources = collect_staged_images(staging)
            if not sources:
                raise RuntimeError(f"TRDG produced no images for job: {job.name}")
            if len(sources) != job.count:
                print(
                    f"[!] {job.name}: requested {job.count}, produced {len(sources)}"
                )

            tasks = []
            for source in sources:
                label = label_from_trdg_filename(source)
                tasks.append((source, output_dir, label, job.pool, final_index, args.seed))
                final_index += 1

            if args.skip_postprocess:
                rows = []
                for source, _, label, pool, index, _ in tasks:
                    filename = f"{label}_{pool}-{index:07d}.jpg"
                    with Image.open(source) as opened:
                        opened.convert("RGB").save(output_dir / filename, quality=95)
                    rows.append((filename, label, pool))
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    rows = list(executor.map(process_one, tasks, chunksize=32))
            all_rows.extend(rows)

        write_manifest(output_dir, all_rows, jobs, args)
        print(f"[+] Generated {len(all_rows):,} images in {output_dir}")
        print(f"[+] Labels: {output_dir / 'labels.tsv'}")


if __name__ == "__main__":
    main()
