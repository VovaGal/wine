import argparse
import os
import ssl
import time
import unicodedata
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ocr_10_epoch import expand_parseq


ssl._create_default_https_context = ssl._create_unverified_context

PARSEQ_HEIGHT = 32
PARSEQ_WIDTH = 128
MIN_MEAN_CONFIDENCE = 0.88
MIN_CHARACTER_CONFIDENCE = 0.55
MIN_LOWER_QUARTILE_CONFIDENCE = 0.75
DEFAULT_CRAFT_CANVAS_SIZE = 2560
DEFAULT_RECOGNITION_PASSES = 3

# Reuse the heavy models for every request handled by one API worker.
_MODEL_CACHE = {}


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
    if allow_multilingual:
        return True
    for char in text:
        if char.isascii():
            continue
        # Permit common typographic punctuation, but not letters from a script
        # that the English detector was not configured to find.
        if unicodedata.category(char).startswith(("P", "Z")):
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
    print(f"[*] Loading custom PARSeq recognizer from {checkpoint_path}...")
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


def main(
    image_path,
    allow_multilingual=False,
    save_debug=False,
    checkpoint_path="weights/parseq_wine_best.pt",
    craft_canvas_size=DEFAULT_CRAFT_CANVAS_SIZE,
    recognition_passes=DEFAULT_RECOGNITION_PASSES,
):
    requested_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hardware selected: {requested_device.type.upper()}")

    parseq_model, detector, device, load_time, cache_hit = initialize_pipeline(
        checkpoint_path=checkpoint_path, device=requested_device
    )
    if cache_hit:
        print("[*] Reusing resident OCR models.")

    detect_started = time.perf_counter()
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    detect_time = time.perf_counter() - detect_started

    if not quads:
        print("[-] No text detected.")
        return []

    debug_dir = Path("debug_crops")
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
    recognize_time = time.perf_counter() - recognize_started

    for text, mean_conf, min_conf, lower_quartile, _ in rejected_predictions:
        print(
            f"[DEBUG] Blocked: {text!r} "
            f"(mean={mean_conf:.2f}, min={min_conf:.2f}, q25={lower_quartile:.2f})"
        )

    print("\n" + "=" * 50)
    print("DETECTED TEXT LINES:")
    print("=" * 50)
    for text in extracted_texts:
        print(f"-> {text}")

    print("\n" + "=" * 50)
    print(" PIPELINE TIMING METRICS:")
    print("=" * 50)
    cache_note = " (cached)" if cache_hit else " (one-time worker startup)"
    print(f"Model Loading: {load_time:.3f} seconds{cache_note}")
    print(f"CRAFT Detect:  {detect_time:.3f} seconds")
    print(
        f"PARSeq OCR:    {recognize_time:.3f} seconds "
        f"({len(extracted_texts)} accepted / {len(detected_crop_indices)} crops, "
        f"{recognition_passes} pass{'es' if recognition_passes != 1 else ''})"
    )
    print(f"Total API Run: {(detect_time + recognize_time):.3f} seconds")
    print("=" * 50)
    return extracted_texts


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
        help="Process every .jpg/.jpeg file directly inside this folder.",
    )
    parser.add_argument(
        "--checkpoint",
        default="weights/parseq_wine_best.pt",
        help="Path to the trained PARSeq checkpoint.",
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
        help="Allow non-ASCII scripts in recognized output.",
    )
    parser.add_argument(
        "--save-debug-crops",
        action="store_true",
        help="Save rectified OCR crops in debug_crops/.",
    )
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
                if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg"}
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
    arguments = parse_arguments()
    image_paths = collect_image_paths(arguments.images, arguments.folder)
    for image_path in image_paths:
        main(
            str(image_path),
            allow_multilingual=arguments.allow_multilingual,
            save_debug=arguments.save_debug_crops,
            checkpoint_path=arguments.checkpoint,
            craft_canvas_size=arguments.craft_canvas_size,
            recognition_passes=arguments.recognition_passes,
        )


## execution tags
# --save-debug-crops                    # check the text boxes detected
# --recognition-passes 1                # 1 max speed, 3 max accuracy
# --folder ".\dataset\ilya_cropped\"    # run through the folder