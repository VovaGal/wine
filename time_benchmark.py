import statistics
import time
from pathlib import Path

import torch
from wine_inference import WineInference

engine = WineInference(
    catalog_path="dataset/wine_catalog.jsonl",
    checkpoint_path="weights/parseq_wine_best.pt",
    device="cuda:0",
)

images = [
    Path("dataset/test/eval_1.webp").read_bytes(),
    Path("dataset/test/eval_2_cropped.jpg").read_bytes(),
    Path("dataset/test/eval_3.jpg").read_bytes(),
]

# Exclude one-time model loading and GPU warm-up from production request timing.
engine.predict(images[0])

times = []
for i in range(30):
    image_bytes = images[i % len(images)]

    torch.cuda.synchronize()
    start = time.perf_counter()
    result = engine.predict(image_bytes, save_debug=False, verbose=False)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)

ordered = sorted(times)
print(f"Median OCR + matching: {statistics.median(times):.3f} s")
print(f"P95 OCR + matching:    {ordered[int(0.95 * (len(ordered) - 1))]:.3f} s")
print(f"Slowest:               {ordered[-1]:.3f} s")