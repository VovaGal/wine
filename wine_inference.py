"""Single-file wine label OCR and catalog inference for backend workers.

    engine = WineInference(catalog_path="dataset/wine_catalog.jsonl",
                           checkpoint_path="weights/parseq_wine_best.pt",
                           device="cuda:0")
    result = engine.predict(image_bytes)  # also accepts a path or OpenCV BGR image

Initialize one instance per worker and reuse it for requests. Keep the generated
wine catalog JSONL and PARSeq checkpoint beside your deployed application.
Scores rank catalog wines; they are not calibrated probabilities.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict, namedtuple
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import cv2
import easyocr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

PARSEQ_HEIGHT = 32
PARSEQ_WIDTH = 128
MIN_MEAN_CONFIDENCE = 0.88
MIN_CHARACTER_CONFIDENCE = 0.55
MIN_LOWER_QUARTILE_CONFIDENCE = 0.75
DEFAULT_CRAFT_CANVAS_SIZE = 2560
DEFAULT_RECOGNITION_PASSES = 3

_MODEL_CACHE = {}


SPACE = re.compile(r"\s+")


def display_text(value: Any) -> str:
    """Keep original spelling while normalizing whitespace and controls."""
    if value is None or isinstance(value, (dict, list, tuple)):
        return ""
    value = unicodedata.normalize("NFC", str(value))
    value = "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in value
    )
    return SPACE.sub(" ", value).strip()


def search_key(value: Any) -> str:
    """Index and query key, shared byte-for-byte with the catalog builder."""
    text = unicodedata.normalize("NFKC", display_text(value)).casefold()
    text = text.replace("ё", "е").replace("’", "'")
    text = "".join(char if char.isalnum() else " " for char in text)
    return SPACE.sub(" ", text).strip()


def accent_key(value: Any) -> str:
    """Fold Latin accents for lookup only; preserve Cyrillic and other scripts."""
    primary = search_key(value)
    if not primary:
        return ""
    # Some letters are not decomposed by NFKD (e.g. ø and ł).
    special = str.maketrans({"æ": "ae", "œ": "oe", "ø": "o", "ł": "l", "đ": "d", "ð": "d", "þ": "th"})
    decomposed = unicodedata.normalize("NFKD", primary.translate(special))
    result: list[str] = []
    last_base_is_latin = False
    for char in decomposed:
        if unicodedata.category(char).startswith("M"):
            if not last_base_is_latin:
                result.append(char)
            continue
        last_base_is_latin = "LATIN" in unicodedata.name(char, "")
        result.append(char)
    return search_key(unicodedata.normalize("NFC", "".join(result)))


def search_forms(value: Any) -> tuple[str, ...]:
    """Deduplicated exact + accent-folded keys in preference order."""
    primary = search_key(value)
    secondary = accent_key(value)
    if not primary:
        return ()
    return (primary, secondary) if secondary and secondary != primary else (primary,)


def tokens(value: Any) -> tuple[str, ...]:
    """Normalized tokens for later field-aware fuzzy matching."""
    key = search_key(value)
    return tuple(key.split()) if key else ()


def script_of(value: Any) -> str:
    """Return latin, cyrillic, mixed, numeric or other."""
    text = display_text(value)
    scripts = {
        script
        for char in text
        if char.isalpha()
        for script in (unicodedata.name(char, "").split(" ", 1)[0],)
        if script in {"LATIN", "CYRILLIC"}
    }
    if len(scripts) == 2:
        return "mixed"
    if "LATIN" in scripts:
        return "latin"
    if "CYRILLIC" in scripts:
        return "cyrillic"
    if any(char.isdigit() for char in text):
        return "numeric"
    return "other"


ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
FIELD_WEIGHTS = {
    "name": 0.65,
    "producer": 0.18,
    "grapes": 0.08,
    "region": 0.04,
    "style": 0.05,
}

# Only unambiguous terms printed on common bilingual labels. These are search
# equivalents, not changes to the original OCR or catalog wording.
EQUIVALENT_WORDS = {
    "sweet": ("сладкое", "сладкий"),
    "brut": ("брют",),
    "dry": ("сухое", "сухой"),
    "rose": ("розе",),
    "розе": ("rose",),
    "брют": ("brut",),
}


OCRLine = namedtuple("OCRLine", "text confidence keys indices")


def parse_ocr_lines(lines: Iterable[str | dict[str, Any]]) -> list[OCRLine]:
    """Accept raw runner output or structured OCR; reject invalid confidences."""
    if isinstance(lines, (str, bytes, dict)):
        raise TypeError("ocr_lines must be a list of strings or {text, confidence} objects")
    parsed: list[OCRLine] = []
    for item in lines:
        if isinstance(item, str):
            value, confidence = item, None
        elif isinstance(item, dict):
            value, confidence = item.get("text"), item.get("confidence")
            if not isinstance(value, str):
                raise ValueError("Every OCR entry must include a text string")
            if confidence is not None:
                if isinstance(confidence, bool) or not isinstance(confidence, (float, int)):
                    raise ValueError("OCR confidence must be a number in [0, 1]")
                confidence = float(confidence)
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("OCR confidence must be a finite number in [0, 1]")
        else:
            raise TypeError("OCR entries must be strings or {text, confidence} objects")
        text = display_text(value)
        keys = search_forms(text)
        if keys:
            parsed.append(OCRLine(text, confidence, keys, (len(parsed),)))
    return parsed


def query_lines(lines: list[OCRLine]) -> list[OCRLine]:
    """Also compare neighboring OCR lines to product names split by CRAFT."""
    result = list(lines)

    def add_group(group: list[OCRLine]) -> None:
        text = " ".join(item.text for item in group)
        # Very long composites are almost always OCR bleed across label
        # regions and cost disproportionately much to compare.
        if len(text) > 42:
            return
        known = [item.confidence for item in group if item.confidence is not None]
        confidence = sum(known) / len(known) if len(known) == len(group) else None
        result.append(OCRLine(text, confidence, search_forms(text), tuple(
            index for item in group for index in item.indices
        )))

    for width in (2, 3):
        for start in range(len(lines) - width + 1):
            add_group(lines[start:start + width])
    # CRAFT can place a stray decorative line between two parts of a name.
    # Skip at most one intermediate line, keeping the search bounded.
    for start in range(len(lines) - 2):
        if min(len(lines[start].text), len(lines[start + 2].text)) < 5:
            continue
        add_group([lines[start], lines[start + 2]])
        if start + 3 < len(lines):
            add_group([lines[start], lines[start + 2], lines[start + 3]])
    return result


def grams(text: str) -> set[str]:
    """Character trigrams from each token, useful for one-letter OCR errors."""
    result: set[str] = set()
    for word in text.split():
        if len(word) >= 4 and ALPHA.search(word):
            result.update(word[index:index + 3] for index in range(len(word) - 2))
    return result


@lru_cache(maxsize=200_000)
def similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    if len(left) > 1.6 * len(right) and len(left.split()) > 1:
        return 0.0
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    ratio = matcher.ratio() if matcher.quick_ratio() >= 0.62 else 0.0
    left_words, right_words = left.split(), right.split()
    # OCR often sees only part of a printed product name. Permit a short OCR
    # line to match a longer catalog title, but NEVER grant the same partial
    # bonus to a longer OCR phrase that merely contains a generic short name.
    if len(left_words) <= len(right_words):
        for size in {len(left_words), min(len(right_words), len(left_words) + 1)}:
            for start in range(len(right_words) - size + 1):
                window = " ".join(right_words[start:start + size])
                if 2 * min(len(left), len(window)) / (len(left) + len(window)) < 0.62:
                    continue
                part = SequenceMatcher(None, left, window, autojunk=False)
                if part.quick_ratio() >= 0.62:
                    ratio = max(ratio, part.ratio() * (
                        0.92 if len(right_words) > size else 1.0
                    ))
    return ratio




def expand_parseq(model, charset, max_label_length, device):
    """Rebuild PARSeq to match a saved custom checkpoint.

    This intentionally lives in the inference runner instead of importing the
    training script. Production inference can therefore load a checkpoint even
    when ocr_10_epoch.py has been renamed, moved, or is not deployed.
    """
    old_tokenizer = model.tokenizer
    tokenizer_class = type(old_tokenizer)
    new_tokenizer = tokenizer_class(charset)

    old_embed = model.model.text_embed.embedding
    old_head = model.model.head
    old_pos = model.model.pos_queries
    max_label_length = int(max_label_length)

    new_embed = nn.Embedding(
        len(new_tokenizer),
        old_embed.embedding_dim,
        device=device,
        dtype=old_embed.weight.dtype,
    )
    nn.init.normal_(new_embed.weight, std=0.02)
    for token, new_id in new_tokenizer._stoi.items():
        old_id = old_tokenizer._stoi.get(token)
        if old_id is not None and old_id < old_embed.num_embeddings:
            new_embed.weight.data[new_id].copy_(old_embed.weight.data[old_id])

    # The classifier predicts EOS plus charset symbols. BOS and PAD occupy the
    # final two tokenizer IDs and are not classifier outputs.
    new_head = nn.Linear(
        old_head.in_features,
        len(new_tokenizer) - 2,
        device=device,
        dtype=old_head.weight.dtype,
    )
    nn.init.normal_(new_head.weight, std=0.02)
    nn.init.zeros_(new_head.bias)
    for token, new_id in new_tokenizer._stoi.items():
        if new_id in (new_tokenizer.bos_id, new_tokenizer.pad_id):
            continue
        old_id = old_tokenizer._stoi.get(token)
        if old_id is not None and old_id < old_head.out_features:
            new_head.weight.data[new_id].copy_(old_head.weight.data[old_id])
            if old_head.bias is not None:
                new_head.bias.data[new_id].copy_(old_head.bias.data[old_id])

    new_pos = nn.Parameter(torch.empty(
        1,
        max_label_length + 1,
        old_pos.shape[-1],
        device=device,
        dtype=old_pos.dtype,
    ))
    nn.init.trunc_normal_(new_pos, std=0.02)
    keep = min(old_pos.shape[1], new_pos.shape[1])
    new_pos.data[:, :keep].copy_(old_pos.data[:, :keep])

    model.model.text_embed.embedding = new_embed
    model.model.head = new_head
    model.model.pos_queries = new_pos
    model.model.max_label_length = max_label_length
    model.tokenizer = new_tokenizer
    model.bos_id = new_tokenizer.bos_id
    model.eos_id = new_tokenizer.eos_id
    model.pad_id = new_tokenizer.pad_id

    if hasattr(model, "hparams"):
        try:
            model.hparams.max_label_length = max_label_length
            model.hparams.charset_train = charset
            model.hparams.charset_test = charset
        except Exception:
            pass


def order_quad(points):
    """Return a quadrilateral as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    differences = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(differences)]
    ordered[3] = pts[np.argmax(differences)]
    return ordered


def horizontal_to_quad(box):
    xmin, xmax, ymin, ymax = map(float, box)
    return np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
        dtype=np.float32,
    )


def quad_bounds(quad):
    quad = np.asarray(quad)
    return (
        float(quad[:, 0].min()),
        float(quad[:, 1].min()),
        float(quad[:, 0].max()),
        float(quad[:, 1].max()),
    )


def box_iou(first, second):
    ax1, ay1, ax2, ay2 = quad_bounds(first)
    bx1, by1, bx2, by2 = quad_bounds(second)
    intersection_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_w * intersection_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def remove_duplicate_boxes(quads, iou_threshold=0.65):
    """Suppress duplicate horizontal/free-form CRAFT detections."""
    ranked = sorted(
        quads,
        key=lambda q: (quad_bounds(q)[2] - quad_bounds(q)[0])
        * (quad_bounds(q)[3] - quad_bounds(q)[1]),
        reverse=True,
    )
    kept = []
    for quad in ranked:
        if not any(box_iou(quad, accepted) >= iou_threshold for accepted in kept):
            kept.append(quad)
    return kept

def joined_fragment_boxes(quads, max_extra=8):
    """Generate extra crops for adjacent and overlapping text fragments."""
    bounds = [quad_bounds(q) for q in quads]
    proposals = []

    for i, (x1, y1, x2, y2) in enumerate(bounds):
        h1 = y2 - y1
        if h1 <= 0:
            continue

        for j, (xx1, yy1, xx2, yy2) in enumerate(bounds):
            if i == j:
                continue

            h2 = yy2 - yy1
            if h2 <= 0:
                continue

            gap = xx1 - x2
            overlap = max(0, min(y2, yy2) - max(y1, yy1))

            if not (
                -0.15 * min(h1, h2) <= gap <= 2.0 * max(h1, h2)
                and overlap >= 0.45 * min(h1, h2)
                and max(h1, h2) / min(h1, h2) <= 1.8
            ):
                continue

            left, right = min(x1, xx1), max(x2, xx2)
            if right - left > 12 * max(h1, h2):
                continue

            joined = horizontal_to_quad((
                left, right, min(y1, yy1), max(y2, yy2)
            ))
            proposals.append((max(h1, h2), gap, joined))

    proposals.sort(key=lambda item: (-item[0], item[1]))

    pairs = []
    pair_limit = max(1, max_extra - 2)
    for _, _, quad in proposals:
        if not any(box_iou(quad, existing) > 0.85 for existing in pairs):
            pairs.append(quad)
        if len(pairs) >= pair_limit:
            break

    result = list(pairs)

    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            ax1, ay1, ax2, ay2 = quad_bounds(pairs[a])
            bx1, by1, bx2, by2 = quad_bounds(pairs[b])

            staggered = (
                ax1 < bx1 < ax2 < bx2
                or bx1 < ax1 < bx2 < ax2
            )
            shared_x = max(0, min(ax2, bx2) - max(ax1, bx1))
            shared_y = max(0, min(ay2, by2) - max(ay1, by1))

            if not (
                staggered
                and shared_x >= 0.25 * min(ax2 - ax1, bx2 - bx1)
                and shared_y >= 0.60 * min(ay2 - ay1, by2 - by1)
            ):
                continue

            joined = horizontal_to_quad((
                min(ax1, bx1), max(ax2, bx2),
                min(ay1, by1), max(ay2, by2),
            ))
            if not any(box_iou(joined, existing) > 0.85 for existing in result):
                result.append(joined)

            if len(result) >= max_extra:
                return result

    return result


def valid_geometry(quad, image_shape):
    """Reject tiny decorative fragments before asking the recognizer to read them."""
    x1, y1, x2, y2 = quad_bounds(quad)
    width = x2 - x1
    height = y2 - y1
    image_h, image_w = image_shape[:2]
    if width < 10 or height < 7 or width * height < 100:
        return False
    if width > image_w * 0.99 and height > image_h * 0.99:
        return False
    ratio = width / max(height, 1.0)
    return 0.35 <= ratio <= 35.0


def get_perspective_crop(img, quad, margin=0.05):
    """Rectify a CRAFT quadrilateral while retaining a small safety margin."""
    points = order_quad(quad)
    centre = points.mean(axis=0)
    points = centre + (points - centre) * (1.0 + 2.0 * margin)
    points[:, 0] = np.clip(points[:, 0], 0, img.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, img.shape[0] - 1)

    tl, tr, br, bl = points
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if width < 2 or height < 2:
        return None

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points.astype(np.float32), destination)
    crop = cv2.warpPerspective(
        img,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if crop.shape[0] > crop.shape[1] * 1.6:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def border_colour(crop):
    border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]), axis=0)
    return tuple(int(value) for value in np.median(border, axis=0))


def letterbox_for_parseq(crop, target_size=(PARSEQ_WIDTH, PARSEQ_HEIGHT)):
    """Fit without changing glyph proportions, then pad to PARSeq's input size."""
    target_w, target_h = target_size
    height, width = crop.shape[:2]
    scale = min(target_w / float(width), target_h / float(height))
    resized_w = max(1, min(target_w, int(round(width * scale))))
    resized_h = max(1, min(target_h, int(round(height * scale))))
    resized = cv2.resize(crop, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((target_h, target_w, 3), border_colour(crop), dtype=np.uint8)
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2
    canvas[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized
    return canvas


def token_confidences(confidence):
    """Normalize tokenizer confidence output across PARSeq hub versions."""
    if isinstance(confidence, torch.Tensor):
        sample = confidence[0] if confidence.ndim > 1 else confidence
    elif isinstance(confidence, (list, tuple)):
        sample = confidence[0] if confidence else []
    else:
        array = np.asarray(confidence)
        sample = array[0] if array.ndim > 1 else array

    # Some PARSeq releases return one CUDA tensor per sample in a Python list.
    # Check the unwrapped sample again before NumPy conversion: CUDA tensors
    # must be detached and copied to CPU explicitly.
    if isinstance(sample, torch.Tensor):
        values = sample.detach().float().cpu().flatten().numpy()
    else:
        values = np.asarray(sample, dtype=np.float32).reshape(-1)
    return values[np.isfinite(values)]


def uses_allowed_script(text, allow_multilingual=False):
    for char in text:
        if char.isascii() or unicodedata.category(char).startswith(("P", "Z", "M")):
            continue
        script_name = unicodedata.name(char, "")
        if "LATIN" in script_name:
            continue
        if allow_multilingual and "CYRILLIC" in script_name:
            continue
        return False
    return True


def plausible_text(text, allow_multilingual=False):
    cleaned = text.strip()
    if len(cleaned) < 2 or cleaned == "100":
        return False
    if not uses_allowed_script(cleaned, allow_multilingual):
        return False
    meaningful = sum(char.isalnum() for char in cleaned)
    if meaningful / max(len(cleaned), 1) < 0.60:
        return False
    # Long runs such as "IIIIII" and punctuation-only logo fragments are
    # frequent confident recognizer hallucinations.
    run = 1
    for previous, current in zip(cleaned, cleaned[1:]):
        run = run + 1 if current.casefold() == previous.casefold() else 1
        if run >= 4:
            return False
    return True


def prediction_is_reliable(text, confidence, allow_multilingual=False):
    values = token_confidences(confidence)
    if not plausible_text(text, allow_multilingual) or values.size == 0:
        return False, 0.0, 0.0, 0.0

    # Exclude any trailing EOS confidence when the decoder exposes more scores
    # than visible characters. Keeping at least the text-length scores makes the
    # rule compatible with both common tokenizer implementations.
    visible_chars = max(1, sum(not char.isspace() for char in text))
    if values.size > visible_chars:
        values = values[:visible_chars]

    mean_conf = float(values.mean())
    min_conf = float(values.min())
    lower_quartile = float(np.quantile(values, 0.25))
    accepted = (
        mean_conf >= MIN_MEAN_CONFIDENCE
        and min_conf >= MIN_CHARACTER_CONFIDENCE
        and lower_quartile >= MIN_LOWER_QUARTILE_CONFIDENCE
    )
    return accepted, mean_conf, min_conf, lower_quartile


def normalize_for_duplicate_check(text):
    return "".join(char.casefold() for char in text if char.isalnum())


def collapse_adjacent_repeated_tokens(text):
    """Turn decoder loops such as '2024 2024' into '2024'."""
    tokens = text.split()
    if not tokens:
        return text
    collapsed = [tokens[0]]
    for token in tokens[1:]:
        previous = normalize_for_duplicate_check(collapsed[-1])
        current = normalize_for_duplicate_check(token)
        if not current or current != previous:
            collapsed.append(token)
    return " ".join(collapsed)


def normalized_edit_similarity(first, second):
    """Character similarity used to select the most stable TTA prediction."""
    first = normalize_for_duplicate_check(first)
    second = normalize_for_duplicate_check(second)
    if first == second:
        return 1.0
    if not first or not second:
        return 0.0

    previous = list(range(len(second) + 1))
    for row, left_char in enumerate(first, start=1):
        current = [row]
        for column, right_char in enumerate(second, start=1):
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            substitution = previous[column - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(len(first), len(second))


def recognition_margins(number_of_passes):
    if number_of_passes == 1:
        return (0.04,)
    if number_of_passes == 3:
        return (0.00, 0.04, 0.08)
    raise ValueError("recognition_passes must be 1 or 3")


def choose_consensus_prediction(records):
    """Choose an accepted prediction using exact votes, then edit consensus."""
    reliable = [record for record in records if record[1]]
    if not reliable:
        return None

    groups = {}
    for record in reliable:
        groups.setdefault(normalize_for_duplicate_check(record[0]), []).append(record)

    largest_group = max(groups.values(), key=lambda group: (len(group), group[0][2]))
    if len(largest_group) >= 2:
        return max(largest_group, key=lambda record: record[2])

    def consensus_score(record):
        similarities = [
            normalized_edit_similarity(record[0], other[0])
            for other in reliable
            if other is not record
        ]
        agreement = float(np.mean(similarities)) if similarities else 1.0
        # Confidence remains primary; agreement breaks close calls and a tiny
        # length penalty discourages a crop from appending neighbouring text.
        return record[2] + 0.08 * agreement - 0.001 * len(record[0])

    return max(reliable, key=consensus_score)


def load_parseq_model(checkpoint_path="weights/parseq_wine_best.pt", device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = torch.hub.load(
        "baudm/parseq", "parseq", pretrained=False, trust_repo=True
    ).to(device)
    expand_parseq(
        model, checkpoint["charset"], checkpoint["max_label_length"], device
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def initialize_pipeline(
    checkpoint_path="weights/parseq_wine_best.pt", device=None, warmup=True
):
    """Load and cache both OCR models once per long-lived API worker."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    cache_key = (os.path.abspath(checkpoint_path), str(device))
    if cache_key in _MODEL_CACHE:
        parseq_model, detector = _MODEL_CACHE[cache_key]
        return parseq_model, detector, device, 0.0, True

    load_started = time.perf_counter()
    if device.type == "cuda":
        # Preserve FP32 recognition accuracy; cuDNN can still tune the fixed
        # detector shapes without reducing arithmetic precision.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True

    parseq_model = load_parseq_model(checkpoint_path, device=device)
    detector = easyocr.Reader(["en"], gpu=(device.type == "cuda"))

    if warmup:
        dummy = torch.zeros(
            (1, 3, PARSEQ_HEIGHT, PARSEQ_WIDTH),
            device=device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            parseq_model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    load_time = time.perf_counter() - load_started
    _MODEL_CACHE[cache_key] = (parseq_model, detector)
    return parseq_model, detector, device, load_time, False




class WineInference:
    """One reusable worker object: CRAFT + PARSeq OCR and catalog matching.

    Construct once per worker. `predict()` accepts a path, encoded image bytes,
    or an OpenCV BGR ndarray. `match()` also accepts OCR lines directly.
    The match_score is an uncalibrated ranking score, not a probability.
    """

    def __init__(
        self,
        catalog_path: str | Path = "dataset/wine_catalog.jsonl",
        checkpoint_path: str | Path = "weights/parseq_wine_best.pt",
        device: str | torch.device | None = None,
        craft_canvas_size: int = DEFAULT_CRAFT_CANVAS_SIZE,
        recognition_passes: int = DEFAULT_RECOGNITION_PASSES,
        allow_multilingual: bool = True,
        warmup: bool = True,
        load_models: bool = True,
    ):
        self.catalog_path = Path(catalog_path).expanduser().resolve()
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {self.device}, but CUDA is unavailable")
        recognition_margins(recognition_passes)
        if craft_canvas_size < 32:
            raise ValueError("craft_canvas_size must be at least 32")
        self.craft_canvas_size = craft_canvas_size
        self.recognition_passes = recognition_passes
        self.allow_multilingual = allow_multilingual
        self.warmup = warmup
        self.parseq_model = None
        self.detector = None
        self.model_load_seconds = 0.0
        self._catalog_signature = None
        self._load_catalog()
        if load_models:
            self.initialize()

    def _load_catalog(self):
        path = self.catalog_path
        signature = (path.stat().st_mtime_ns, path.stat().st_size)
        if signature == self._catalog_signature:
            return 0.0
        started = time.perf_counter()
        self.products: dict[str, dict[str, Any]] = {}
        with self.catalog_path.open("r", encoding="utf-8-sig") as handle:
            for row_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    product = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid catalog JSON on line {row_number}") from exc
                if product.get("schema_version") != 1:
                    raise ValueError(f"Unsupported catalog schema on line {row_number}")
                wine_id = str(product.get("wine_id", ""))
                if not wine_id or wine_id in self.products:
                    raise ValueError(f"Missing or duplicate wine_id on line {row_number}")
                self.products[wine_id] = product
        if not self.products:
            raise ValueError(f"Empty wine catalog: {self.catalog_path}")

        self.producer_ids = {}
        for wine_id, product in self.products.items():
            key = search_key(product.get("producer"))
            if key:
                self.producer_ids.setdefault(key, []).append(wine_id)

        self.fields: dict[str, dict[str, tuple[str, ...]]] = {}
        self.alias_owners: dict[str, set[str]] = defaultdict(set)
        self.token_index: dict[str, set[str]] = defaultdict(set)
        self.gram_index: dict[str, set[str]] = defaultdict(set)
        for wine_id, product in self.products.items():
            aliases = [item["text"] for item in product.get("aliases", [])]
            fields = {
                "name": self._unique_forms(aliases or [product.get("name")]),
                "producer": self._unique_forms([product.get("producer")]),
                "grapes": self._unique_forms(product.get("grapes") or []),
                "region": self._unique_forms([product.get("region")]),
                "style": self._unique_forms(product.get("style") or []),
            }
            self.fields[wine_id] = fields
            for alias in fields["name"]:
                self.alias_owners[alias].add(wine_id)
            # Only identifying fields retrieve candidates; style/region often
            # appear on hundreds of unrelated wines.
            for field in ("name", "producer", "grapes"):
                for value in fields[field]:
                    for word in tokens(value):
                        if len(word) >= 3 and ALPHA.search(word):
                            self.token_index[word].add(wine_id)
                    for gram in grams(value):
                        self.gram_index[gram].add(wine_id)
        self._catalog_signature = signature
        return time.perf_counter() - started

    def initialize(self) -> float:
        """Load and warm up OCR models once; call during worker startup."""
        if self.parseq_model is not None:
            return 0.0
        self.parseq_model, self.detector, _, elapsed, _ = initialize_pipeline(
            self.checkpoint_path, self.device, self.warmup
        )
        self.model_load_seconds = elapsed
        return elapsed

    @staticmethod
    def _unique_forms(values: Iterable[Any]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(form for value in values for form in search_forms(value)))

    def specificity(self, key: str) -> float:
        """Common tokens (e.g. wine/riserva) should not identify a product."""
        corpus_size = len(self.products)
        word_scores = []
        for word in key.split():
            if len(word) < 3 or not ALPHA.search(word):
                continue
            appearances = len(self.token_index.get(word, ()))
            if not appearances:
                # An OCR typo absent from the catalog is not proof of rarity.
                continue
            value = math.log1p(corpus_size / (1 + appearances)) / math.log1p(corpus_size)
            word_scores.append(value)
        return max(0.35, max(word_scores, default=0.35))

    def retrieve(self, queries: list[OCRLine], limit: int = 80) -> list[str]:
        votes: Counter[str] = Counter()
        corpus_size = len(self.products)
        for query in queries:
            # Numeric-only lines are very common. They can help rerank only
            # when the catalog actually knows a vintage.
            if not any(ALPHA.search(key) for key in query.keys):
                continue
            per_line: Counter[str] = Counter()
            for key in query.keys:
                for word in set(key.split()):
                    if len(word) < 3 or not ALPHA.search(word):
                        continue
                    matches = self.token_index.get(word, ())
                    if matches:
                        weight = math.log1p(corpus_size / (1 + len(matches)))
                        per_line.update({wine_id: weight for wine_id in matches})
                for gram in grams(key):
                    matches = self.gram_index.get(gram, ())
                    if matches:
                        weight = 0.17 * math.log1p(corpus_size / (1 + len(matches)))
                        per_line.update({wine_id: weight for wine_id in matches})
            for wine_id, vote in sorted(
                per_line.items(), key=lambda pair: (-pair[1], pair[0])
            )[:35]:
                votes[wine_id] += vote
        return [wine_id for wine_id, _ in sorted(votes.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]

    def _best_evidence(
        self, queries: list[OCRLine], values: tuple[str, ...], field: str
    ) -> dict[str, Any] | None:
        best: tuple[float, OCRLine, str, float] | None = None
        for query in queries:
            for query_key in query.keys:
                for candidate in values:
                    likeness = similarity(query_key, candidate)
                    if likeness < 0.62:
                        continue
                    confidence_factor = (
                        1.0 if query.confidence is None
                        else 0.65 + 0.35 * query.confidence
                    )
                    # Measure rarity on the CATALOG alias: an OCR misspelling
                    # such as FRIZEN must not erase the rarity of FROZEN.
                    specificity = self.specificity(candidate) if field == "name" else 1.0
                    # A common single word appearing inside a long title is
                    # weak evidence even when that title contains rare words.
                    phrase_coverage = (
                        min(1.0, len(query_key.split()) / len(candidate.split()))
                        if field in {"name", "producer"} else 1.0
                    )
                    score = likeness * confidence_factor * specificity * phrase_coverage
                    if best is None or score > best[0]:
                        best = (score, query, candidate, likeness)
        if best is None:
            return None
        score, query, candidate, likeness = best
        return {
            "field": field,
            "ocr_text": query.text,
            "catalog_key": candidate,
            "ocr_indices": list(query.indices),
            "similarity": round(likeness, 3),
            "weighted_similarity": round(score, 3),
            "contribution": round(score * FIELD_WEIGHTS[field], 4),
        }

    def _title_token_evidence(
        self, lines: list[OCRLine], wine_id: str
    ) -> dict[str, Any] | None:
        """Recognize a catalog title printed as several separate OCR lines."""
        observations: dict[str, tuple[int, float]] = {}
        for index, line in enumerate(lines):
            for key in line.keys:
                for word in key.split():
                    if len(word) < 3 and not (len(word) >= 2 and word.isdigit()):
                        continue
                    confidence = line.confidence if line.confidence is not None else 1.0
                    if word not in observations or confidence > observations[word][1]:
                        observations[word] = (index, confidence)
                    for alternate in EQUIVALENT_WORDS.get(word, ()):
                        observations.setdefault(alternate, (index, confidence * 0.92))
        if not observations:
            return None
        best = None
        for alias in self.fields[wine_id]["name"]:
            words = [
                word for word in alias.split()
                if len(word) >= 3 or (len(word) >= 2 and word.isdigit())
            ]
            if not words:
                continue
            total_weight = 0.0
            supported_weight = 0.0
            matched_indices: set[int] = set()
            for word in words:
                weight = max(0.35, self.specificity(word))
                total_weight += weight
                best_word = max(
                    (
                        (similarity(observed, word), index, confidence)
                        for observed, (index, confidence) in observations.items()
                        if 2 * min(len(observed), len(word)) / (len(observed) + len(word)) >= 0.75
                    ),
                    default=(0.0, -1, 0.0),
                )
                likeness, index, confidence = best_word
                if likeness >= 0.78:
                    supported_weight += weight * likeness * (0.65 + 0.35 * confidence)
                    matched_indices.add(index)
            coverage = supported_weight / total_weight
            # The OCR may have read a single common term in an otherwise
            # unrelated title. Require a substantial share of this title.
            if coverage < 0.58 or not matched_indices:
                continue
            evidence = {
                "field": "name",
                "ocr_text": " | ".join(lines[index].text for index in sorted(matched_indices)),
                "catalog_key": alias,
                "ocr_indices": sorted(matched_indices),
                "similarity": round(coverage, 3),
                "weighted_similarity": round(coverage, 3),
                "contribution": round(coverage * FIELD_WEIGHTS["name"], 4),
                "matched_words": len(matched_indices),
            }
            if best is None or evidence["weighted_similarity"] > best["weighted_similarity"]:
                best = evidence
        return best

    def rank(
        self, wine_id: str, queries: list[OCRLine], lines: list[OCRLine],
        rare_words: set[str],
    ) -> dict[str, Any]:
        product = self.products[wine_id]
        evidence = []
        score = 0.0
        used_indices: set[int] = set()
        for field, weight in FIELD_WEIGHTS.items():
            item = self._best_evidence(queries, self.fields[wine_id][field], field)
            if field == "name":
                spread = self._title_token_evidence(lines, wine_id)
                if spread and (not item or spread["weighted_similarity"] > item["weighted_similarity"]):
                    item = spread
            if item:
                # The same OCR line is one piece of evidence. A bottle with
                # "Riesling" in both its name and grape field should not get
                # a second boost from that single printed word.
                indices = set(item["ocr_indices"])
                if indices.issubset(used_indices):
                    if field == "producer" and search_key(product.get("producer")) in self.fields[wine_id]["name"][0]:
                        # A printed brand can occur inside the product name.
                        # Give modest producer credit even when both point at
                        # the same OCR crop.
                        score += 0.10 * item["weighted_similarity"]
                    continue
                evidence.append(item)
                score += weight * item["weighted_similarity"]
                if field == "name" and "matched_words" in item:
                    # Resolve ties between a complete printed name and its
                    # shorter generic sibling (e.g. Cantiani Riesling vs.
                    # Cantiani Aligote Riesling).
                    score += min(0.08, 0.02 * item["matched_words"])
                used_indices.update(indices)

        # Vintage is supporting evidence only if the catalog has an explicit
        # year. In the supplied scrape, every year is null.
        year = product.get("year")
        if isinstance(year, int) and not isinstance(year, bool):
            for query in queries:
                if YEAR.search(query.text) and str(year) in YEAR.findall(query.text):
                    score += 0.03
                    evidence.append({
                        "field": "year", "ocr_text": query.text,
                        "catalog_key": str(year), "ocr_indices": list(query.indices),
                        "similarity": 1.0, "weighted_similarity": 1.0,
                        "contribution": 0.03,
                    })
                    break
        known_words = {
            word for field in ("name", "producer", "grapes")
            for value in self.fields[wine_id][field]
            for word in value.split()
        }
        missing_rare = [
            word for word in rare_words
            if not any(similarity(word, known) >= 0.85 for known in known_words)
        ]
        # A rare, correctly read brand on the label must count against an
        # unrelated wine that merely shares its varietal name.
        score = max(0.0, score - min(0.30, 0.12 * len(missing_rare)))
        return {
            "wine_id": wine_id,
            "name": product["name"],
            "producer": product.get("producer"),
            "image_url": product.get("image_url"),
            "match_score": round(min(1.0, score), 4),
            "evidence": evidence,
        }

    @staticmethod
    def evidence_coverage(lines: list[OCRLine], evidence: list[dict[str, Any]]) -> float:
        """Fraction of descriptive OCR lines explained by identifying fields."""
        descriptive = [
            index for index, line in enumerate(lines)
            if any(len(word) >= 4 and ALPHA.search(word) for word in line.keys[0].split())
        ]
        if not descriptive:
            return 0.0
        matched = {
            index for item in evidence
            if item["field"] in {"name", "producer", "grapes"}
            and item["similarity"] >= 0.78
            for index in item["ocr_indices"]
        }
        return len(matched.intersection(descriptive)) / len(descriptive)

    def match(self, ocr_lines: Iterable[str | dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._load_catalog()
        # If its better to show the producer shows itself first, then this block should be replaced with the commented
        # however that increase the total time by about 1s, 0.4s median - benchmarked
        parsed = parse_ocr_lines(ocr_lines)
        source = [{"text": line.text, "confidence": line.confidence} for line in parsed]
        result = {
            "status": "unresolved",
            "ocr_lines": source,
            "best_match": None,
            "alternatives": [],
            "candidates": [],
            "score_type": "uncalibrated_catalog_match_score",
        }
        if not parsed:
            return result

        # parsed = parse_ocr_lines(ocr_lines)
        # source = [
        #     {"text": line.text, "confidence": line.confidence}
        #     for line in parsed
        # ]
        
        # result = {
        #     "status": "unresolved",
        #     "ocr_lines": source,
        #     "best_match": None,
        #     "alternatives": [],
        #     "candidates": [],
        #     "score_type": "uncalibrated_catalog_match_score",
        #     "recognized_producer": None,
        #     "producer_wines": [],
        # }
        
        # if not parsed:
        #     return result
        
        # producer_key = next(
        #     (
        #         key
        #         for line in parsed
        #         if line.confidence is None or line.confidence >= 0.80
        #         for key in line.keys
        #         if len(key) >= 5 and key in self.producer_ids
        #     ),
        #     None,
        # )
        
        # if producer_key:
        #     ids = self.producer_ids[producer_key]
        #     result["recognized_producer"] = self.products[ids[0]]["producer"]
        #     result["producer_wines"] = [
        #         {
        #             "wine_id": wine_id,
        #             "name": self.products[wine_id]["name"],
        #             "slug": self.products[wine_id].get("slug"),
        #             "image_url": self.products[wine_id].get("image_url"),
        #         }
        #         for wine_id in ids
        #     ]

        queries = query_lines(parsed)
        ids = self.retrieve(queries)
        if not ids:
            return result
        rare_words = {
            word for line in parsed for key in line.keys for word in key.split()
            if len(word) >= 5 and 0 < len(self.token_index.get(word, ())) <= 100
        }
        ranked = sorted(
            (self.rank(wine_id, queries, parsed, rare_words) for wine_id in ids),
            key=lambda item: (-item["match_score"], item["wine_id"]),
        )
        ranked = ranked[:top_k]
        first = ranked[0]
        first["evidence_coverage"] = round(
            self.evidence_coverage(parsed, first["evidence"]), 3
        )
        runner_up = ranked[1]["match_score"] if len(ranked) > 1 else 0.0
        name_evidence = next(
            (e for e in first["evidence"] if e["field"] == "name"), None
        )
        alias_owners = (
            self.alias_owners.get(name_evidence["catalog_key"], set())
            if name_evidence else set()
        )
        producer_supported = any(
            e["field"] == "producer" and e["similarity"] >= 0.85
            for e in first["evidence"]
        )
        name_identifies_product = bool(
            name_evidence and len(alias_owners) == 1
            and len(name_evidence["catalog_key"]) >= 8
            and name_evidence["weighted_similarity"] >= 0.80
        )
        unique_exact_alias = bool(
            name_evidence
            and name_evidence["catalog_key"] in search_forms(name_evidence["ocr_text"])
            and len(name_evidence["catalog_key"]) >= 6
            and self.specificity(name_evidence["catalog_key"]) >= 0.60
            and alias_owners == {first["wine_id"]}
        )
        unique_fuzzy_alias = False
        if name_evidence:
            alias = name_evidence["catalog_key"]
            ocr_key = search_key(name_evidence["ocr_text"])
            # Full phrase, similar length, and a rare token that survives a
            # one-character OCR error. A partial match on "Ruby" or "Blend"
            # does not qualify as a bottle identity.
            rare_token_seen = any(
                len(word) >= 5
                and 0 < len(self.token_index.get(word, ())) <= 3
                and any(similarity(typed, word) >= 0.75 for typed in ocr_key.split())
                for word in alias.split()
            )
            unique_fuzzy_alias = (
                len(alias) >= 8
                and len(ocr_key) >= 8
                and len(alias.split()) == len(ocr_key.split())
                and name_evidence["similarity"] >= 0.85
                and alias_owners == {first["wine_id"]}
                and rare_token_seen
            )
        if first["match_score"] >= 0.52 and name_evidence and (name_identifies_product or producer_supported) and (
            first["match_score"] - runner_up >= 0.08 or unique_exact_alias or unique_fuzzy_alias
        ) and (first["evidence_coverage"] >= 0.65 or unique_exact_alias or unique_fuzzy_alias) and (
            len(parsed) > 1 or name_evidence["weighted_similarity"] >= 0.75
        ):
            result["status"] = "matched"
        elif first["match_score"] >= 0.40 and name_evidence and first["evidence_coverage"] >= 0.60:
            result["status"] = "ambiguous"
        # Two catalog SKUs may have identical name and producer but different
        # grapes, which are not printed on the observed front label.
        if (
            result["status"] == "unresolved"
            and producer_supported and len(ranked) > 1
            and search_key(first["name"]) == search_key(ranked[1]["name"])
            and search_key(first["producer"]) == search_key(ranked[1]["producer"])
            and first["match_score"] >= 0.55
        ):
            result["status"] = "ambiguous"
        if len(alias_owners) >= 3 and not producer_supported:
            result["status"] = "unresolved"
        # A bare grape or style, without a brand, describes dozens of wines.
        near_top = sum(
            candidate["match_score"] >= first["match_score"] - 0.10
            for candidate in ranked
        )
        if (
            result["status"] == "ambiguous"
            and near_top >= min(5, top_k)
            and not any(e["field"] == "producer" for e in first["evidence"])
        ):
            result["status"] = "unresolved"
        if result["status"] == "unresolved":
            result["rejection_reason"] = (
                "No sufficiently supported product match in this catalog; "
                "check whether the label's wine is present."
            )
            return result
        result["best_match"] = first
        result["alternatives"] = ranked[1:]
        result["candidates"] = ranked
        return result

    def predict(
        self,
        image: str | os.PathLike | bytes | np.ndarray,
        *,
        allow_multilingual: bool | None = None,
        save_debug: bool = False,
        craft_canvas_size: int | None = None,
        recognition_passes: int | None = None,
        top_k: int = 5,
        debug_dir: str | Path = "debug_crops",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Return OCR lines, wine candidates and measured timings as a JSON-ready dict."""
        image_label = str(image) if isinstance(image, (str, os.PathLike)) else None
        if recognition_passes is None:
            recognition_passes = self.recognition_passes
        if craft_canvas_size is None:
            craft_canvas_size = self.craft_canvas_size
        if allow_multilingual is None:
            allow_multilingual = self.allow_multilingual
        recognition_margins(recognition_passes)
        if craft_canvas_size < 32:
            raise ValueError("craft_canvas_size must be at least 32")
        catalog_load_time = self._load_catalog()
        decode_started = time.perf_counter()
        if isinstance(image, (bytes, bytearray, memoryview)):
            image_bgr = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image, np.ndarray):
            image_bgr = image
        elif isinstance(image, (str, os.PathLike)):
            image_bgr = cv2.imread(str(image))
        else:
            raise TypeError("image must be a path, encoded image bytes, or BGR ndarray")
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Image is unreadable or is not a 3-channel BGR image")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        decode_time = time.perf_counter() - decode_started
        load_time = self.initialize()
        parseq_model, detector, device = self.parseq_model, self.detector, self.device
        detect_started = time.perf_counter()

        horizontal_groups, free_groups = detector.detect(
            image_rgb,
            text_threshold=0.72,
            low_text=0.40,
            link_threshold=0.40,
            mag_ratio=1.5,
            canvas_size=craft_canvas_size,
            slope_ths=0.2,
            add_margin=0.02,
        )
        horizontal_boxes = horizontal_groups[0] if horizontal_groups else []
        free_boxes = free_groups[0] if free_groups else []

        quads = [horizontal_to_quad(box) for box in horizontal_boxes]
        quads.extend(order_quad(poly) for poly in free_boxes if len(poly) == 4)
        quads = [quad for quad in quads if valid_geometry(quad, image_rgb.shape)]
        quads = remove_duplicate_boxes(quads)
        quads.sort(key=lambda quad: (quad_bounds(quad)[1], quad_bounds(quad)[0]))

        original_crop_count = len(quads)
        quads.extend(joined_fragment_boxes(quads))
        joined_crop_count = len(quads) - original_crop_count

        detect_time = time.perf_counter() - detect_started

        debug_dir = Path(debug_dir)
        if save_debug:
            debug_dir.mkdir(parents=True, exist_ok=True)

        tensor_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        batch_tensors = []
        batch_metadata = []
        detected_crop_indices = set()
        margins = recognition_margins(recognition_passes)
        for crop_index, quad in enumerate(quads):
            for pass_index, margin in enumerate(margins):
                crop = get_perspective_crop(image_rgb, quad, margin=margin)
                if crop is None:
                    continue
                prepared = letterbox_for_parseq(crop)
                batch_tensors.append(tensor_transform(Image.fromarray(prepared)))
                batch_metadata.append((crop_index, pass_index))
                detected_crop_indices.add(crop_index)
                if save_debug:
                    cv2.imwrite(
                        str(debug_dir / f"crop_{crop_index:02d}_pass_{pass_index}.png"),
                        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                    )

        recognize_started = time.perf_counter()
        accepted_predictions = []
        rejected_predictions = []
        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)
            with torch.inference_mode():
                logits = parseq_model(batch)
            probabilities = logits.softmax(-1)
            with torch.inference_mode():
                labels, confidences = parseq_model.tokenizer.decode(probabilities)

            candidates_by_crop = {}
            for index, raw_text in enumerate(labels):
                text = collapse_adjacent_repeated_tokens(raw_text.strip())
                confidence = confidences[index : index + 1]
                accepted, mean_conf, min_conf, lower_quartile = prediction_is_reliable(
                    text, confidence, allow_multilingual
                )
                crop_index, pass_index = batch_metadata[index]
                # Candidate: text, accepted, mean, minimum, lower quartile,
                # detection order, augmentation pass.
                record = (
                    text,
                    accepted,
                    mean_conf,
                    min_conf,
                    lower_quartile,
                    crop_index,
                    pass_index,
                )
                candidates_by_crop.setdefault(crop_index, []).append(record)

            for crop_index in sorted(candidates_by_crop):
                candidates = candidates_by_crop[crop_index]
                selected = choose_consensus_prediction(candidates)
                if selected is not None:
                    accepted_predictions.append(
                        (selected[0], selected[2], selected[3], selected[4], crop_index)
                    )
                else:
                    best_rejected = max(candidates, key=lambda record: record[2])
                    rejected_predictions.append(
                        (
                            best_rejected[0],
                            best_rejected[2],
                            best_rejected[3],
                            best_rejected[4],
                            crop_index,
                        )
                    )

        # Suppress duplicate readings caused by nested/overlapping regions while
        # retaining the most confident instance and restoring top-to-bottom order.
        best_by_text = {}
        for record in accepted_predictions:
            key = normalize_for_duplicate_check(record[0])
            if key and (key not in best_by_text or record[1] > best_by_text[key][1]):
                best_by_text[key] = record
        accepted_predictions = sorted(best_by_text.values(), key=lambda item: item[4])
        extracted_texts = [record[0] for record in accepted_predictions]
        ocr_lines = [
            {"text": record[0], "confidence": record[1]}
            for record in accepted_predictions
        ]
        recognize_time = time.perf_counter() - recognize_started

        match_started = time.perf_counter()
        result = self.match(ocr_lines, top_k=top_k)
        match_time = time.perf_counter() - match_started

        if verbose:
            print(f"Image: {image_label}")
            print("Detected text:", extracted_texts)
            print("Match status:", result["status"])
            for rank, candidate in enumerate(result["candidates"], 1):
                print(f"{rank}. {candidate['name']} | score {candidate['match_score']:.4f} | ID {candidate['wine_id']}")
            for text, mean_conf, min_conf, q25, _ in rejected_predictions:
                print(f"[DEBUG] Blocked: {text!r} (mean={mean_conf:.2f}, min={min_conf:.2f}, q25={q25:.2f})")
            print(f"CRAFT: {detect_time:.3f}s; PARSeq: {recognize_time:.3f}s; match: {match_time:.3f}s")
            print(f"Joined crops: {joined_crop_count}; "
                f"debug indices start at {original_crop_count:02d}")

        result["image_path"] = image_label
        result["timing_seconds"] = {
            "model_loading": round(load_time, 4),
            "catalog_loading": round(catalog_load_time, 4),
            "image_decode": round(decode_time, 4),
            "craft_detect": round(detect_time, 4),
            "parseq_ocr": round(recognize_time, 4),
            "catalog_match": round(match_time, 4),
            "request_total": round(decode_time + detect_time + recognize_time + match_time + catalog_load_time, 4),
        }
        return result


def parse_arguments():
    parser = argparse.ArgumentParser(description="CRAFT + custom PARSeq label OCR")
    parser.add_argument(
        "images",
        nargs="*",
        default=[],
        help="One or more label images; models are reused between them.",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        help="Process every .jpg/.jpeg/.png file directly inside this folder.",
    )
    parser.add_argument(
        "--checkpoint",
        default="weights/parseq_wine_best.pt",
        help="Path to the trained PARSeq checkpoint.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("dataset/wine_catalog.jsonl"),
        help="Catalog JSONL generated by build_wine_catalog.py.",
    )
    parser.add_argument(
        "--craft-canvas-size",
        type=int,
        default=DEFAULT_CRAFT_CANVAS_SIZE,
        help="CRAFT working resolution; 2560 is the accuracy-first default.",
    )
    parser.add_argument(
        "--recognition-passes",
        type=int,
        choices=(1, 3),
        default=DEFAULT_RECOGNITION_PASSES,
        help="Use 3 crop variants for accuracy or 1 for maximum speed.",
    )
    parser.add_argument(
        "--allow-multilingual",
        action="store_true",
        default=True,
        help="Accept Latin and Cyrillic text (already the default).",
    )
    parser.add_argument(
        "--latin-only",
        dest="allow_multilingual",
        action="store_false",
        help="Reject Cyrillic predictions; accented Latin remains supported.",
    )
    parser.add_argument(
        "--save-debug-crops",
        action="store_true",
        help="Save rectified OCR crops in debug_crops/.",
    )
    parser.add_argument("--device", default=None, help="OCR device (cpu, cuda, cuda:0).")
    parser.add_argument("--top-k", type=int, default=5, help="Number of catalog matches to return.")
    return parser.parse_args()


def collect_image_paths(images, folder=None):
    paths = [Path(image) for image in images]
    if folder is not None:
        if not folder.is_dir():
            raise NotADirectoryError(f"Image folder does not exist: {folder}")
        paths.extend(
            sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png"}
            )
        )
    if not paths:
        paths = [Path("chianti.jpg")]

    # Preserve order while preventing the same explicit/folder image from
    # being processed twice.
    unique_paths = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
    return unique_paths


if __name__ == "__main__":
    args = parse_arguments()
    engine = WineInference(
        catalog_path=args.catalog,
        checkpoint_path=args.checkpoint,
        device=args.device,
        craft_canvas_size=args.craft_canvas_size,
        recognition_passes=args.recognition_passes,
        allow_multilingual=args.allow_multilingual,
    )
    for path in collect_image_paths(args.images, args.folder):
        result = engine.predict(str(path), save_debug=args.save_debug_crops, top_k=args.top_k, verbose=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
