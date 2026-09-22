# pip install ultralytics opencv-python
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLOWorld
from concurrent.futures import ThreadPoolExecutor

def extract_label(img_path, output_dir, model):
    img = cv2.imread(str(img_path))
    if img is None:
        return
        
    # YOLO-World automatically searches for our custom text prompt
    results = model.predict(img, conf=0.1, verbose=False)
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        # Get the bounding box with the highest confidence
        boxes = results[0].boxes
        best_box = boxes[boxes.conf.argmax()]
        
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        
        # Add a slight 5px padding to ensure edge text isn't clipped
        h, w = img.shape[:2]
        x1, y1 = max(0, x1 - 5), max(0, y1 - 5)
        x2, y2 = min(w, x2 + 5), min(h, y2 + 5)
        
        cropped_label = img[y1:y2, x1:x2]
        
        out_name = output_dir / f"{img_path.stem}_label.png"
        cv2.imwrite(str(out_name), cropped_label)
        print(f"Extracted: {out_name.name}")
    else:
        print(f"[!] No label found in: {img_path.name}")

def batch_extract():
    in_dir = Path("dataset/images")
    out_dir = Path("dataset/cropped_labels")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize YOLOv8s-World (downloads a ~50MB weights file automatically)
    model = YOLOWorld('yolov8s-world.pt')
    model.set_classes(["wine label"])
    
    # Process sequentially or in batches (YOLO handles GPU dispatch internally)
    for img_file in in_dir.glob("*.png"):
        extract_label(img_file, out_dir, model)

if __name__ == "__main__":
    batch_extract()