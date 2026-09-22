import json
import re
import cv2
import albumentations as A
from pathlib import Path

# Paths
LABELS_DIR = Path("dataset/cropped_labels")
METADATA_FILE = Path("dataset/wines_metadata.json")
SYNTH_DIR = Path("dataset/synthetic_dataset")
MANIFEST_FILE = SYNTH_DIR / "synthetic_manifest.json"

SYNTH_DIR.mkdir(parents=True, exist_ok=True)

# Upgraded pipeline from previous step
synthesis_pipeline = A.Compose([
    # 1. BOTTLE CURVATURE (Warping)
    # GridDistortion pulls and pushes pixels, perfectly mimicking how a 
    # flat label stretches across the curved glass of a cylinder.
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.7),
    A.Perspective(scale=(0.04, 0.1), keep_size=True, p=0.5),
    
    # 2. OBLIQUE LIGHTING & SHADOWS
    # Casts gradient shadows across the label (simulating a bottle blocking 
    # overhead supermarket lights).
    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, 
                   shadow_dimension=4, shadow_roi=(0, 0, 1, 1), p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    
    # 3. HARSH GLARES & REFLECTIONS
    # Creates bright, washed-out streaks simulating directional light hitting glass.
    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, 
                     num_flare_circles_lower=0, src_radius=100, p=0.3),
                     
    # 4. FINGERS & EDGE OCCLUSION
    # Drops black/flesh-toned rectangles on the edges of the image to 
    # simulate a thumb or hand partially covering the text.
    A.CoarseDropout(max_holes=2, max_height=60, max_width=40, 
                    fill_value=0, p=0.4), # Black occlusion
                    
    # 5. SENSOR NOISE & FOCUS
    A.MotionBlur(blur_limit=7, p=0.4),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.5)

])

def run_synthesis(variants_per_label: int = 5):
    # 1. Load existing scraped metadata into an ID-lookup dict
    wine_db = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            for item in json.load(f):
                wine_db[str(item.get("scraped_id"))] = item

    synthetic_records = []
    label_files = list(LABELS_DIR.glob("*.png"))
    print(f"[*] Found {len(label_files)} cropped labels. Generating {variants_per_label} variants each...")

    for img_path in label_files:
        # Extract numeric ID using regex (e.g., 'wine_108_label.png' -> '108')
        match = re.search(r"wine_(\d+)", img_path.stem)
        wine_id = match.group(1) if match else img_path.stem

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        parent_meta = wine_db.get(wine_id, {})

        # Save baseline clean crop into synthetic folder
        base_filename = f"wine_{wine_id}_base.jpg"
        base_out_path = SYNTH_DIR / base_filename
        cv2.imwrite(str(base_out_path), img)

        synthetic_records.append({
            "image_path": str(base_out_path),
            "wine_id": wine_id,
            "variant_type": "baseline",
            "name": parent_meta.get("name"),
            "producer": parent_meta.get("producer"),
            "year": parent_meta.get("year"),
            "grape": parent_meta.get("grape")
        })

        # Generate augmented variations
        for i in range(variants_per_label):
            augmented = synthesis_pipeline(image=img_rgb)
            aug_bgr = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)

            variant_filename = f"wine_{wine_id}_synth_{i}.jpg"
            variant_out_path = SYNTH_DIR / variant_filename
            cv2.imwrite(str(variant_out_path), aug_bgr)

            synthetic_records.append({
                "image_path": str(variant_out_path),
                "wine_id": wine_id,
                "variant_type": f"synthetic_{i}",
                "name": parent_meta.get("name"),
                "producer": parent_meta.get("producer"),
                "year": parent_meta.get("year"),
                "grape": parent_meta.get("grape")
            })

    # 2. Save the master index linking every image to its wine metadata
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(synthetic_records, f, ensure_ascii=False, indent=2)

    print(f"[+] Generation complete.")
    print(f"[+] Total images indexed: {len(synthetic_records)}")
    print(f"[+] Manifest file saved to: {MANIFEST_FILE}")

if __name__ == "__main__":
    run_synthesis(variants_per_label=5)