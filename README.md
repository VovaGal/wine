# Wine label OCR and catalog matching

The inference runner detects label text with EasyOCR/CRAFT, reads it with the custom PARSeq checkpoint, and matches the OCR lines against a local wine catalog. It returns the best catalog entry and up to four alternatives. The catalog matcher uses Python's standard library; no vector database is needed for this step.

## Files to put in the container

| File | Purpose |
| --- | --- |
| `5_run_dewrap.py` | CLI and backend entry point; detection, recognition, confidence filtering and catalog lookup. Contains `expand_parseq`, so the training script is not required. |
| `wine_matcher.py` | Fuzzy product lookup and ranking; returns up to five catalog candidates. |
| `wine_normalizer.py` | Shared Unicode normalization for the catalog and OCR query. |
| `requirements.txt` | Python inference dependencies. |
| `weights/parseq_wine_best.pt` | **Required input:** trained custom recognizer checkpoint; supply from your trained artifacts. |
| `dataset/wine_catalog.jsonl` | **Required input:** built wine catalog; copy or mount the file you generated. |

For catalog generation only: `build_wine_catalog.py` converts `dataset/wines_metadata.json` into `dataset/wine_catalog.jsonl`. `verified_wines.example.json` shows the format for optional manually verified additions. The scraped metadata, synthetic data, fonts, and training scripts are unnecessary for inference once the checkpoint and catalog exist.

## Install and prepare Docker

Use Python 3.11 and run commands from the directory containing the scripts. Choose a matching `torch` / `torchvision` wheel pair for your container's CUDA runtime (or CPU), then install the remaining packages. Example for **CUDA 12.4**:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

For a CPU container, replace `cu124` with `cpu`. For another CUDA runtime, follow the [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/) and keep the version pair in `requirements.txt` aligned with your chosen wheels. The container also needs a compatible NVIDIA driver and GPU access for CUDA; the runner falls back to CPU when CUDA is unavailable.

**Prepare models before serving traffic:** `torch.hub.load("baudm/parseq", ...)` fetches the PARSeq source into the Torch Hub cache the first time it runs; EasyOCR fetches its detector weights into `~/.EasyOCR/model` if missing. These files are *not* included by `pip install`. Prepopulate both caches in the container image or mount them so a production worker can start without outbound internet. The custom `.pt` checkpoint must also be present at the path above. Keep caches readable by the account running the backend.

If you need to rebuild the catalog from scraped metadata:

```bash
python build_wine_catalog.py --input dataset/wines_metadata.json --output dataset/wine_catalog.jsonl
```

Use `--additions dataset/verified_wines.json` for verified bottles missing from the scrape, and `--overwrite` when intentionally replacing an existing catalog.

## Run inference

```bash
python 5_run_dewrap.py path/to/label.jpg
python 5_run_dewrap.py label1.jpg label2.png
python 5_run_dewrap.py --folder path/to/images
```

`--folder` scans `.jpg`, `.jpeg`, and `.png` directly inside the folder (not subfolders). Multiple images share one in-process model and catalog cache. With no image or folder, the runner tries `chianti.jpg` in the current directory.

| Option | Effect |
| --- | --- |
| `--checkpoint PATH` | Custom PARSeq checkpoint; default `weights/parseq_wine_best.pt`. |
| `--catalog PATH` | Catalog JSONL; default `dataset/wine_catalog.jsonl`. |
| `--recognition-passes {1,3}` | Default `3` for three crop variants; `1` trades recognition accuracy for speed. |
| `--craft-canvas-size N` | CRAFT working resolution; default `2560`. Smaller values may run faster but miss small text. |
| `--latin-only` | Reject Cyrillic OCR predictions; accented Latin is supported. |
| `--allow-multilingual` | Accept Latin and Cyrillic; already the default. |
| `--save-debug-crops` | Write crops to `debug_crops/` for inspection. |

For example, with custom paths and faster recognition:

```bash
python 5_run_dewrap.py --checkpoint /models/parseq_wine_best.pt --catalog /data/wine_catalog.jsonl --recognition-passes 1 /data/label.jpg
```

## Backend integration

Keep the Python worker alive across requests. As `5_run_dewrap.py` starts with a digit, copy it as `wine_inference.py` in the container if your backend wants to import it normally. Keep `wine_matcher.py` and `wine_normalizer.py` alongside it:

```python
import wine_inference

CHECKPOINT = "/models/parseq_wine_best.pt"
CATALOG = "/data/wine_catalog.jsonl"

# Once during worker startup; subsequent main() calls reuse these in process.
wine_inference.initialize_matcher(CATALOG)
wine_inference.initialize_pipeline(CHECKPOINT)

def recognize_wine(image_path: str) -> dict:
    return wine_inference.main(
        image_path,
        checkpoint_path=CHECKPOINT,
        catalog_path=CATALOG,
    )
```

`main()` returns a dictionary with `status` (`matched`, `ambiguous` or `unresolved`), `ocr_lines`, `best_match`, `alternatives`, `candidates`, `score_type`, `image_path` and `timing_seconds`. A candidate contains `wine_id`, `name`, `producer`, `image_url`, `match_score` and matching `evidence`. When the match is unresolved there is no best match or alternatives; do not treat OCR character confidence as product identity confidence. `match_score` ranks catalog entries **but is not a calibrated probability**. The runner prints human-readable diagnostics to stdout as well as returning the dict.

Cache lifetime is per Python process. `model_loading` and `catalog_loading` occur once per worker; `request_total` measures detection + recognition + matching after setup, and is not an end-to-end HTTP or embeddings latency measurement. Provision GPU memory for each backend worker that loads the models.
