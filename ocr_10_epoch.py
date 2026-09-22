import os
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHARSET = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÜÝÞß"
    "àáâãäåæçèéêëìíîïðñòóôõöøùúüýþÿ"
)

# None = automatically use the longest label found in the dataset.
# Set this to an integer (e.g. 96) only if you deliberately want a cap.
MAX_LABEL_LENGTH = None

# Longer sequences are much more expensive in PARSeq's decoder.
# 16 is a safer starting point for an 8 GB GPU than the previous 64.
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 3e-4
EPOCHS = 10
VAL_FRACTION = 0.10
SEED = 42
CHECKPOINT_EVERY = 10


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WineTextDataset(Dataset):
    def __init__(self, data_dir, transform=None, charset=CHARSET, max_label_length=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.charset = set(charset)
        self.max_label_length = max_label_length
        self.samples = []
        self.removed_char_counts = Counter()

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        print(f"[*] Scanning {self.data_dir} for images...")

        image_files = sorted(
            p for p in self.data_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if not image_files:
            raise RuntimeError(f"No JPG/JPEG/PNG files found in {self.data_dir}")

        skipped_bad_name = 0
        skipped_empty = 0
        skipped_too_long = 0

        for filename in image_files:
            base_name = filename.stem

            # TRDG normally writes: LabelText_index.jpg
            # The final underscore is the generated image index.
            if "_" not in base_name:
                skipped_bad_name += 1
                continue

            raw_label = base_name.rsplit("_", 1)[0]

            # Match PARSeq's own CharsetAdapter behavior: remove unsupported
            # characters rather than crashing the tokenizer.
            label_chars = []
            for ch in raw_label:
                if ch in self.charset:
                    label_chars.append(ch)
                else:
                    self.removed_char_counts[ch] += 1
            label = "".join(label_chars)

            if not label:
                skipped_empty += 1
                continue

            if self.max_label_length is not None and len(label) > self.max_label_length:
                skipped_too_long += 1
                continue

            self.samples.append((filename.name, label))

        if not self.samples:
            raise RuntimeError("No valid training samples were found.")

        lengths = [len(label) for _, label in self.samples]
        self.max_observed_length = max(lengths)

        print(f"[*] Found {len(image_files)} image files.")
        print(f"[*] Accepted {len(self.samples)} training images.")
        print(f"[*] Shortest label: {min(lengths)} chars")
        print(f"[*] Longest label:  {max(lengths)} chars")
        print(f"[*] Mean label length: {sum(lengths) / len(lengths):.1f} chars")

        if skipped_too_long:
            print(
                f"[!] Skipped {skipped_too_long} images because they exceed "
                f"the configured max label length of {self.max_label_length}."
            )
        if skipped_bad_name:
            print(f"[!] Skipped {skipped_bad_name} files without the expected '_' separator.")
        if skipped_empty:
            print(f"[!] Skipped {skipped_empty} files whose label became empty.")

        if self.removed_char_counts:
            examples = " ".join(
                f"{repr(ch)}×{count}" for ch, count in self.removed_char_counts.most_common(12)
            )
            print(f"[!] Unsupported filename characters removed: {examples}")

        # Print a compact length distribution so you can immediately see how
        # long the generated labels actually are.
        bins = [(1, 10), (11, 20), (21, 25), (26, 32), (33, 40), (41, 64), (65, 96), (97, 9999)]
        distribution = []
        for lo, hi in bins:
            n = sum(lo <= x <= hi for x in lengths)
            if n:
                label = f"{lo}-{hi}" if hi != 9999 else f">={lo}"
                distribution.append(f"{label}: {n}")
        print("[*] Label-length distribution: " + ", ".join(distribution))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        image = Image.open(self.data_dir / img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# PARSeq vocabulary + positional-query expansion
# ---------------------------------------------------------------------------

def expand_parseq(model, charset, max_label_length, device):
    """Expand PARSeq's charset and decoder length without discarding useful
    pretrained weights that already exist for Latin characters/punctuation."""

    old_tokenizer = model.tokenizer
    TokenizerClass = type(old_tokenizer)
    new_tokenizer = TokenizerClass(charset)

    old_embed = model.model.text_embed.embedding
    old_head = model.model.head
    old_pos = model.model.pos_queries
    old_max_length = int(model.model.max_label_length)

    print(f"[*] Original vocabulary: {len(old_tokenizer)} tokens")
    print(f"[*] New vocabulary:      {len(new_tokenizer)} tokens")
    print(f"[*] Original max label length: {old_max_length}")
    print(f"[*] New max label length:      {max_label_length}")

    # --- Text embedding -----------------------------------------------------
    new_embed = nn.Embedding(
        len(new_tokenizer),
        old_embed.embedding_dim,
        device=device,
        dtype=old_embed.weight.dtype,
    )
    nn.init.normal_(new_embed.weight, std=0.02)

    copied_embed = 0
    for token, new_id in new_tokenizer._stoi.items():
        old_id = old_tokenizer._stoi.get(token)
        if old_id is not None and old_id < old_embed.num_embeddings:
            new_embed.weight.data[new_id].copy_(old_embed.weight.data[old_id])
            copied_embed += 1

    # --- Output classifier --------------------------------------------------
    # PARSeq's head excludes BOS and PAD from prediction. EOS is row 0.
    new_head = nn.Linear(
        old_head.in_features,
        len(new_tokenizer) - 2,
        device=device,
        dtype=old_head.weight.dtype,
    )
    nn.init.normal_(new_head.weight, std=0.02)
    nn.init.zeros_(new_head.bias)

    copied_head = 0
    for token, new_id in new_tokenizer._stoi.items():
        if token in (new_tokenizer.bos_id, new_tokenizer.pad_id):
            continue
        old_id = old_tokenizer._stoi.get(token)
        if old_id is not None and old_id < old_head.out_features:
            new_head.weight.data[new_id].copy_(old_head.weight.data[old_id])
            if old_head.bias is not None:
                new_head.bias.data[new_id].copy_(old_head.bias.data[old_id])
            copied_head += 1

    # --- Positional queries -------------------------------------------------
    # Pretrained PARSeq has 26 positions because max_label_length=25.
    # Keep those pretrained queries and randomly initialize any extra ones.
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

    # Keep the system tokenizer and IDs synchronized.
    model.tokenizer = new_tokenizer
    model.bos_id = new_tokenizer.bos_id
    model.eos_id = new_tokenizer.eos_id
    model.pad_id = new_tokenizer.pad_id

    # The Lightning system uses hparams for reconstruction/documentation.
    if hasattr(model, "hparams"):
        try:
            model.hparams.max_label_length = max_label_length
            model.hparams.charset_train = charset
            model.hparams.charset_test = charset
        except Exception:
            pass

    print(f"[*] Preserved {copied_embed} pretrained embedding rows.")
    print(f"[*] Preserved {copied_head} pretrained classifier rows.")
    print(f"[*] Preserved {keep} pretrained position queries; initialized {max_label_length + 1 - keep} new ones.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scaler, epoch, best_val_loss):
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "charset": CHARSET,
        "max_label_length": int(model.model.max_label_length),
        "img_size": (32, 128),
    }
    torch.save(checkpoint, path)


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    batches = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=(device.type == "cuda"))
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            loss = model.training_step((images, list(labels)), batch_idx=batch_idx)
            if isinstance(loss, dict):
                loss = loss["loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss at batch {batch_idx}: {loss.item()}"
            )

        total_loss += float(loss.detach().item())
        batches += 1

    model.train()
    return total_loss / max(1, batches)


def train_model():
    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "dataset" / "ocr_training" / "dated_synth_crops"
    weights_dir = project_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    if use_amp:
        print(f"[*] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[!] CUDA is not available; training on CPU.")

    print("[*] Loading pretrained PARSeq...")
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True).to(device)

    # Dataset first: determine the actual required sequence length from the
    # labels, instead of throwing away valid Cyrillic samples at length 25.
    print("[*] Loading synthetic dataset metadata...")
    dataset_probe = WineTextDataset(
        data_dir=dataset_dir,
        transform=None,
        max_label_length=MAX_LABEL_LENGTH,
    )

    max_label_length = (
        dataset_probe.max_observed_length
        if MAX_LABEL_LENGTH is None
        else MAX_LABEL_LENGTH
    )

    if max_label_length < 1:
        raise RuntimeError("No usable labels found.")

    # The actual PARSeq model supports configurable max_label_length; the
    # pretrained checkpoint simply starts with 25. Expand its position table.
    print("[*] Expanding PARSeq for the dataset's label length and charset...")
    expand_parseq(model, CHARSET, max_label_length, device)
    model.train()

    # Freeze the ViT encoder. The decoder/heads/position queries remain trainable.
    frozen_prefixes = (
        "model.encoder.patch_embed",
        "model.encoder.pos_embed",
        "model.encoder.cls_token",
        "model.encoder.blocks",
    )
    for name, param in model.named_parameters():
        if name.startswith(frozen_prefixes):
            param.requires_grad = False

    # Keep the pretrained image size. PARSeq's standard input is 32x128.
    img_transform = transforms.Compose([
        transforms.Resize(
            (32, 128),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Reuse the already-scanned labels but attach the image transform.
    dataset = dataset_probe
    dataset.transform = img_transform

    # Hold out 10% for validation. The validation set is never used for updates.
    val_size = max(1, int(len(dataset) * VAL_FRACTION))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    common_loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": use_amp,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, **common_loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, **common_loader_kwargs
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(
        f"[*] Starting fine-tuning on {len(train_dataset)} train + "
        f"{len(val_dataset)} validation images for {EPOCHS} epochs..."
    )
    print(f"[*] Training max label length: {max_label_length}")
    print(f"[*] Batch size: {BATCH_SIZE}")
    print(f"[*] Best checkpoint criterion: validation loss")

    # training_step() calls Lightning's logging methods; we're driving the loop.
    model.log = lambda *args, **kwargs: None
    model.log_dict = lambda *args, **kwargs: None

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=use_amp)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = model.training_step((images, list(labels)), batch_idx=batch_idx)
                if isinstance(loss, dict):
                    loss = loss["loss"]

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch + 1}, batch {batch_idx}: {loss.item()}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach().item())
            epoch_loss += loss_value
            pbar.set_postfix(loss=f"{loss_value:.4f}")

        train_loss = epoch_loss / max(1, len(train_loader))
        val_loss = evaluate_model(model, val_loader, device)

        print(
            f"[+] Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )

        # Always keep the most recent state so an interrupted run can be inspected/resumed.
        save_checkpoint(
            weights_dir / "parseq_wine_last.pt",
            model, optimizer, scaler, epoch + 1, best_val_loss
        )

        # This is the model you should normally use for inference.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            save_checkpoint(
                weights_dir / "parseq_wine_best.pt",
                model, optimizer, scaler, best_epoch, best_val_loss
            )
            print(
                f"[+] NEW BEST -> epoch {best_epoch}, "
                f"val_loss={best_val_loss:.4f}"
            )

        # Keep a historical snapshot every 10 epochs.
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            periodic_path = weights_dir / f"parseq_wine_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                periodic_path, model, optimizer, scaler, epoch + 1, best_val_loss
            )
            print(f"[*] Periodic checkpoint saved: {periodic_path}")

    print(
        f"[+] Training finished. Best epoch: {best_epoch}; "
        f"best validation loss: {best_val_loss:.4f}"
    )


if __name__ == "__main__":
    train_model()