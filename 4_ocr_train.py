"""Fine-tune PARSeq for multilingual wine-label OCR.

Designed for the dataset produced by ocr_synthetic_training_crops.py:
  * labels.tsv is the source of truth (filename parsing remains a fallback)
  * labels are capped at 32 characters
  * repeated renderings of one label never cross the train/validation boundary
  * crops are aspect-ratio padded rather than stretched to 128x32
  * the ViT encoder is warmed up frozen, then its final blocks are fine-tuned
  * best-model selection and early stopping use real OCR CER/exact match

Example:
    python ocr_10_epoch.py --dataset-dir dataset/ocr_training/dated_synth_crops
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm


# PARSeq's pretrained characters remain first so their rows can be copied.
# Extra characters observed in labels.tsv are appended deterministically.
BASE_CHARSET = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÜÝÞß"
    "àáâãäåæçèéêëìíîïðñòóôõöøùúüýþÿ"
    "’°№ŒœŠšŽž"
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_LABEL_LENGTH = 32
IMAGE_SIZE = (32, 128)  # height, width


@dataclass(frozen=True)
class Sample:
    filename: str
    label: str
    pool: str


@dataclass
class Metrics:
    cer: float
    exact_match: float
    casefold_exact_match: float
    mean_confidence: float
    samples: int
    by_pool: dict[str, dict[str, float]]


class ResizePadNormalize:
    """Letterbox to 128x32 using the crop border colour, matching inference."""

    def __init__(self, size: tuple[int, int] = IMAGE_SIZE):
        self.height, self.width = size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5,), (0.5,))

    @staticmethod
    def border_colour(image: Image.Image) -> tuple[int, int, int]:
        array = np.asarray(image.convert("RGB"))
        border = np.concatenate([
            array[0, :, :], array[-1, :, :], array[:, 0, :], array[:, -1, :]
        ])
        return tuple(int(value) for value in np.median(border, axis=0))

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        source_width, source_height = image.size
        if source_width < 1 or source_height < 1:
            raise ValueError("Cannot transform an empty image")

        scale = min(self.width / source_width, self.height / source_height)
        resized_width = max(1, min(self.width, round(source_width * scale)))
        resized_height = max(1, min(self.height, round(source_height * scale)))
        resized = image.resize(
            (resized_width, resized_height), resample=Image.Resampling.BICUBIC
        )
        canvas = Image.new("RGB", (self.width, self.height), self.border_colour(image))
        left = (self.width - resized_width) // 2
        top = (self.height - resized_height) // 2
        canvas.paste(resized, (left, top))
        return self.normalize(self.to_tensor(canvas))


class WineTextDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        transform=None,
        max_label_length: int = MAX_LABEL_LENGTH,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_label_length = max_label_length
        self.samples: list[Sample] = []

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        manifest = self.data_dir / "labels.tsv"
        if manifest.is_file():
            print(f"[*] Reading labels from {manifest}")
            candidates = self._read_manifest(manifest)
        else:
            print("[!] labels.tsv not found; falling back to filename labels.")
            candidates = self._read_filenames()

        skipped_missing = 0
        skipped_empty = 0
        skipped_long = 0
        skipped_control = 0
        duplicate_files = 0
        seen_files: set[str] = set()

        for candidate in candidates:
            filename = Path(candidate.filename).name
            label = unicodedata.normalize("NFC", candidate.label).strip()
            pool = candidate.pool.strip() or "unknown"

            if filename in seen_files:
                duplicate_files += 1
                continue
            seen_files.add(filename)
            if not (self.data_dir / filename).is_file():
                skipped_missing += 1
                continue
            if not label:
                skipped_empty += 1
                continue
            if any(unicodedata.category(char).startswith("C") for char in label):
                skipped_control += 1
                continue
            if len(label) > max_label_length:
                skipped_long += 1
                continue
            self.samples.append(Sample(filename, label, pool))

        if not self.samples:
            raise RuntimeError("No valid training samples were found")

        lengths = [len(sample.label) for sample in self.samples]
        pool_counts = Counter(sample.pool for sample in self.samples)
        print(f"[*] Accepted {len(self.samples):,} training images")
        print(
            f"[*] Label lengths: min={min(lengths)}, max={max(lengths)}, "
            f"mean={sum(lengths) / len(lengths):.1f}"
        )
        print("[*] Pools: " + ", ".join(f"{k}={v:,}" for k, v in sorted(pool_counts.items())))
        if any((skipped_missing, skipped_empty, skipped_long, skipped_control, duplicate_files)):
            print(
                "[!] Skipped: "
                f"missing={skipped_missing}, empty={skipped_empty}, "
                f"over_length={skipped_long}, control_chars={skipped_control}, "
                f"duplicate_files={duplicate_files}"
            )

    @staticmethod
    def _read_manifest(path: Path) -> list[Sample]:
        rows: list[Sample] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"filename", "label", "pool"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path} must contain columns: filename, label, pool")
            for row in reader:
                rows.append(Sample(row["filename"], row["label"], row["pool"]))
        return rows

    def _read_filenames(self) -> list[Sample]:
        rows: list[Sample] = []
        for path in sorted(self.data_dir.iterdir()):
            if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
                continue
            if "_" not in path.stem:
                continue
            label, suffix = path.stem.rsplit("_", 1)
            pool = suffix.split("-", 1)[0] if "-" in suffix else "unknown"
            rows.append(Sample(path.name, label, pool))
        return rows

    @property
    def labels(self) -> list[str]:
        return [sample.label for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(self.data_dir / sample.filename) as opened:
            image = opened.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label, sample.pool


def make_charset(labels: Iterable[str]) -> str:
    observed = {char for label in labels for char in unicodedata.normalize("NFC", label)}
    charset = list(dict.fromkeys(BASE_CHARSET))
    known = set(charset)
    charset.extend(sorted(observed - known, key=ord))
    return "".join(charset)


def grouped_stratified_split(
    dataset: WineTextDataset, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split whole label groups, approximately preserving each pool ratio."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        groups[sample.label.casefold()].append(index)

    by_pool: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for key, indices in groups.items():
        by_pool[dataset.samples[indices[0]].pool].append((key, indices))

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for pool, pool_groups in sorted(by_pool.items()):
        rng.shuffle(pool_groups)
        target = max(1, round(sum(len(indices) for _, indices in pool_groups) * val_fraction))
        selected = 0
        for group_number, (_, indices) in enumerate(pool_groups):
            groups_left = len(pool_groups) - group_number
            should_validate = selected < target and groups_left > 1
            if should_validate:
                val_indices.extend(indices)
                selected += len(indices)
            else:
                train_indices.extend(indices)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    if not train_indices or not val_indices:
        raise RuntimeError("Could not form non-empty grouped train/validation splits")

    train_labels = {dataset.samples[i].label.casefold() for i in train_indices}
    val_labels = {dataset.samples[i].label.casefold() for i in val_indices}
    overlap = train_labels & val_labels
    if overlap:
        raise RuntimeError(f"Label leakage detected across split: {next(iter(overlap))!r}")
    print(
        f"[*] Grouped split: train={len(train_indices):,} images/"
        f"{len(train_labels):,} labels, val={len(val_indices):,} images/"
        f"{len(val_labels):,} labels"
    )
    return train_indices, val_indices


def expand_parseq(model, charset: str, max_label_length: int, device: torch.device) -> None:
    """Expand vocabulary/positions while preserving compatible pretrained rows."""
    old_tokenizer = model.tokenizer
    tokenizer_class = type(old_tokenizer)
    new_tokenizer = tokenizer_class(charset)

    old_embed = model.model.text_embed.embedding
    old_head = model.model.head
    old_pos = model.model.pos_queries
    old_max_length = int(model.model.max_label_length)

    print(f"[*] Original vocabulary: {len(old_tokenizer)} tokens")
    print(f"[*] New vocabulary:      {len(new_tokenizer)} tokens")
    print(f"[*] Original max label length: {old_max_length}")
    print(f"[*] New max label length:      {max_label_length}")

    new_embed = nn.Embedding(
        len(new_tokenizer), old_embed.embedding_dim,
        device=device, dtype=old_embed.weight.dtype,
    )
    nn.init.normal_(new_embed.weight, std=0.02)
    copied_embed = 0
    for token, new_id in new_tokenizer._stoi.items():
        old_id = old_tokenizer._stoi.get(token)
        if old_id is not None and old_id < old_embed.num_embeddings:
            new_embed.weight.data[new_id].copy_(old_embed.weight.data[old_id])
            copied_embed += 1

    new_head = nn.Linear(
        old_head.in_features, len(new_tokenizer) - 2,
        device=device, dtype=old_head.weight.dtype,
    )
    nn.init.normal_(new_head.weight, std=0.02)
    nn.init.zeros_(new_head.bias)
    copied_head = 0
    for token, new_id in new_tokenizer._stoi.items():
        if new_id in (new_tokenizer.bos_id, new_tokenizer.pad_id):
            continue
        old_id = old_tokenizer._stoi.get(token)
        if old_id is not None and old_id < old_head.out_features:
            new_head.weight.data[new_id].copy_(old_head.weight.data[old_id])
            if old_head.bias is not None:
                new_head.bias.data[new_id].copy_(old_head.bias.data[old_id])
            copied_head += 1

    new_pos = nn.Parameter(torch.empty(
        1, max_label_length + 1, old_pos.shape[-1],
        device=device, dtype=old_pos.dtype,
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

    print(f"[*] Preserved {copied_embed} pretrained embedding rows")
    print(f"[*] Preserved {copied_head} pretrained classifier rows")
    print(f"[*] Preserved {keep} position queries; initialized {max_label_length + 1 - keep}")


def freeze_encoder(model) -> None:
    for parameter in model.model.encoder.parameters():
        parameter.requires_grad = False


def unfreeze_encoder_tail(model, block_count: int) -> int:
    encoder = model.model.encoder
    blocks = getattr(encoder, "blocks", None)
    if blocks is None or len(blocks) == 0:
        print("[!] Encoder blocks were not found; encoder remains frozen")
        return 0
    for block in blocks[-block_count:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    norm = getattr(encoder, "norm", None)
    if norm is not None:
        for parameter in norm.parameters():
            parameter.requires_grad = True
    return sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad)


def edit_distance(reference: str, hypothesis: str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, hyp_char in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (ref_char != hyp_char),
            ))
        previous = current
    return previous[-1]


@torch.inference_mode()
def evaluate_model(model, dataloader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    total_edits = 0
    total_characters = 0
    exact = 0
    casefold_exact = 0
    confidence_sum = 0.0
    confidence_count = 0
    sample_count = 0
    pool_stats: dict[str, Counter] = defaultdict(Counter)

    for images, labels, pools in tqdm(dataloader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=(device.type == "cuda"))
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            probabilities = model(images).softmax(-1)
        predictions, confidences = model.tokenizer.decode(probabilities)

        for target, prediction, pool, confidence in zip(labels, predictions, pools, confidences):
            distance = edit_distance(target, prediction)
            target_length = max(1, len(target))
            total_edits += distance
            total_characters += target_length
            is_exact = prediction == target
            is_casefold_exact = prediction.casefold() == target.casefold()
            exact += int(is_exact)
            casefold_exact += int(is_casefold_exact)
            sample_count += 1
            pool_stats[pool]["edits"] += distance
            pool_stats[pool]["characters"] += target_length
            pool_stats[pool]["exact"] += int(is_exact)
            pool_stats[pool]["samples"] += 1

            if isinstance(confidence, torch.Tensor) and confidence.numel():
                confidence_sum += float(confidence.float().mean().item())
                confidence_count += 1

    by_pool = {}
    for pool, stats in sorted(pool_stats.items()):
        by_pool[pool] = {
            "cer": stats["edits"] / max(1, stats["characters"]),
            "exact_match": stats["exact"] / max(1, stats["samples"]),
            "samples": int(stats["samples"]),
        }
    return Metrics(
        cer=total_edits / max(1, total_characters),
        exact_match=exact / max(1, sample_count),
        casefold_exact_match=casefold_exact / max(1, sample_count),
        mean_confidence=confidence_sum / max(1, confidence_count),
        samples=sample_count,
        by_pool=by_pool,
    )


def split_fingerprint(dataset: WineTextDataset, indices: Iterable[int]) -> str:
    labels = sorted(dataset.samples[index].label.casefold() for index in indices)
    return hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest()


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    scheduler,
    epoch: int,
    best_cer: float,
    best_exact_match: float,
    charset: str,
    metrics: Metrics,
    split_info: dict,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_cer": best_cer,
        "best_exact_match": best_exact_match,
        "metrics": {
            "cer": metrics.cer,
            "exact_match": metrics.exact_match,
            "casefold_exact_match": metrics.casefold_exact_match,
            "mean_confidence": metrics.mean_confidence,
            "by_pool": metrics.by_pool,
        },
        "charset": charset,
        "max_label_length": int(model.model.max_label_length),
        "img_size": IMAGE_SIZE,
        "preprocessing": "aspect_ratio_letterbox_border_median",
        "split": split_info,
    }
    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset/ocr_training/dated_synth_crops")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum epochs; early stopping may finish sooner")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--encoder-warmup-epochs", type=int, default=2)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_model() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers non-negative")
    if not 0.01 <= args.val_fraction <= 0.40:
        raise ValueError("--val-fraction must be between 0.01 and 0.40")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    weights_dir = Path(args.weights_dir).expanduser().resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[*] Device: {torch.cuda.get_device_name(0) if use_amp else 'CPU'}")

    dataset = WineTextDataset(
        dataset_dir, transform=ResizePadNormalize(), max_label_length=MAX_LABEL_LENGTH
    )
    charset = make_charset(dataset.labels)
    train_indices, val_indices = grouped_stratified_split(
        dataset, args.val_fraction, args.seed
    )
    split_info = {
        "seed": args.seed,
        "method": "casefolded_label_grouped_pool_stratified",
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "train_fingerprint": split_fingerprint(dataset, train_indices),
        "validation_fingerprint": split_fingerprint(dataset, val_indices),
    }

    print("[*] Loading pretrained PARSeq")
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True).to(device)
    expand_parseq(model, charset, MAX_LABEL_LENGTH, device)
    freeze_encoder(model)

    model.log = lambda *args, **kwargs: None
    model.log_dict = lambda *args, **kwargs: None

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    loader_common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": use_amp,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_common["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_subset, shuffle=True, drop_last=True, **loader_common
    )
    val_loader = DataLoader(
        val_subset, shuffle=False, drop_last=False, **loader_common
    )

    encoder_parameters = list(model.model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW([
        {"params": decoder_parameters, "lr": args.decoder_lr, "name": "decoder"},
        {"params": encoder_parameters, "lr": args.encoder_lr, "name": "encoder"},
    ], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(
        f"[*] Training {len(train_subset):,} + validating {len(val_subset):,} "
        f"samples for at most {args.epochs} epochs"
    )
    print(f"[*] Batch size: {args.batch_size}; max label length: {MAX_LABEL_LENGTH}")
    print(
        f"[*] Encoder frozen for {args.encoder_warmup_epochs} epochs, then final "
        f"{args.unfreeze_blocks} blocks are fine-tuned"
    )

    best_cer = math.inf
    best_exact = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.encoder_warmup_epochs + 1:
            count = unfreeze_encoder_tail(model, args.unfreeze_blocks)
            print(f"[*] Unfroze encoder tail: {count:,} trainable parameters")

        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_index, (images, labels, _) in enumerate(progress):
            images = images.to(device, non_blocking=use_amp)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = model.training_step((images, list(labels)), batch_idx=batch_index)
                if isinstance(loss, dict):
                    loss = loss["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}: {loss.item()}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            active_parameters = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(active_parameters, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            value = float(loss.detach().item())
            running_loss += value
            progress.set_postfix(loss=f"{value:.4f}")

        train_loss = running_loss / max(1, len(train_loader))
        metrics = evaluate_model(model, val_loader, device)
        scheduler.step(metrics.cer)
        learning_rates = ", ".join(
            f"{group.get('name', 'group')}={group['lr']:.2e}"
            for group in optimizer.param_groups
        )
        print(
            f"[+] Epoch {epoch}: loss={train_loss:.4f}, CER={metrics.cer:.4f}, "
            f"exact={metrics.exact_match:.2%}, casefold_exact="
            f"{metrics.casefold_exact_match:.2%}, confidence={metrics.mean_confidence:.3f}"
        )
        print(f"    LR: {learning_rates}")
        print("    Pools: " + ", ".join(
            f"{pool} CER={values['cer']:.3f}/exact={values['exact_match']:.1%}"
            for pool, values in metrics.by_pool.items()
        ))

        improved = (
            metrics.cer < best_cer - 1e-5
            or (abs(metrics.cer - best_cer) <= 1e-5 and metrics.exact_match > best_exact)
        )
        if improved:
            best_cer = metrics.cer
            best_exact = metrics.exact_match
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            weights_dir / "parseq_wine_last.pt", model, optimizer, scaler,
            scheduler, epoch, best_cer, best_exact, charset, metrics, split_info,
        )
        if improved:
            save_checkpoint(
                weights_dir / "parseq_wine_best.pt", model, optimizer, scaler,
                scheduler, epoch, best_cer, best_exact, charset, metrics, split_info,
            )
            print(
                f"[+] NEW BEST: epoch {epoch}, CER={best_cer:.4f}, "
                f"exact={best_exact:.2%}"
            )
        if epoch % 5 == 0:
            save_checkpoint(
                weights_dir / f"parseq_wine_epoch_{epoch:03d}.pt",
                model, optimizer, scaler, scheduler, epoch,
                best_cer, best_exact, charset, metrics, split_info,
            )

        if (
            epoch >= args.min_epochs
            and epochs_without_improvement >= args.patience
        ):
            print(
                f"[*] Early stopping: no CER improvement for "
                f"{epochs_without_improvement} epochs"
            )
            break

    print(
        f"[+] Finished. Best epoch={best_epoch}, CER={best_cer:.4f}, "
        f"exact match={best_exact:.2%}"
    )
    print(f"[+] Use {weights_dir / 'parseq_wine_best.pt'} for inference")


if __name__ == "__main__":
    train_model()
