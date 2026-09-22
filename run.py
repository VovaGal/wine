import sys
import ssl
import cv2
import torch
import easyocr
from PIL import Image
from torchvision import transforms
import os
import time

os.makedirs("debug_crops", exist_ok=True)

# Bypass SSL verification to allow EasyOCR to download its detector weights
ssl._create_default_https_context = ssl._create_unverified_context

# Import the architecture logic directly from your training script
from ocr_10_epoch import expand_parseq

def load_parseq_model(checkpoint_path="weights/parseq_wine_best.pt", device="cuda"):
    print(f"[*] Loading custom PARSeq recognizer from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=False, trust_repo=True).to(device)
    
    # Reconstruct the model size using the saved parameters
    expand_parseq(model, checkpoint["charset"], checkpoint["max_label_length"], device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def main(image_path):
    t_start = time.perf_counter()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hardware selected: {device.type.upper()}")
    
    # 1. Load Models
    t0 = time.perf_counter()
    parseq_model = load_parseq_model(device=device)
    detector = easyocr.Reader(['en'], gpu=(device.type=='cuda'))
    t_load = time.perf_counter() - t0
    
    # 2. Image Prep & Detection
    t1 = time.perf_counter()
    img = cv2.imread(image_path)
    if img is None:
        print(f"[-] Error: Could not read {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = detector.detect(img_rgb)
    horizontal_boxes = result[0][0] if result[0] else []
    t_detect = time.perf_counter() - t1
    
    if not horizontal_boxes:
        print("[-] No text detected.")
        return

    # 3. Recognition Loop (Batched for Speed)
    t2 = time.perf_counter()
    img_transform = transforms.Compose([
        transforms.Resize((32, 128), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    horizontal_boxes = sorted(horizontal_boxes, key=lambda b: b[2])
    extracted_texts = []
    batch_tensors = []
    
    for box in horizontal_boxes:
        x_min, x_max, y_min, y_max = [int(v) for v in box]
        crop_np = img_rgb[max(0, y_min):y_max, max(0, x_min):x_max]
        
        if crop_np.size == 0:
            continue
            
        crop_tensor = img_transform(Image.fromarray(crop_np))
        batch_tensors.append(crop_tensor)
        
    if batch_tensors:
        # Stack all crops into a single tensor [N, 3, 32, 128] and pass to GPU ONCE
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        with torch.inference_mode():
            logits = parseq_model(batch_tensor)
            # Decode the entire batch simultaneously
            for i in range(logits.shape[0]):
                # Add batch dimension back for the tokenizer decode method
                single_logit = logits[i].unsqueeze(0).softmax(-1)
                label, _ = parseq_model.tokenizer.decode(single_logit)
                extracted_texts.append(label[0])
            
    t_recognize = time.perf_counter() - t2
    t_total = time.perf_counter() - t_start

    # --- PRINT RESULTS & METRICS ---
    print("\n" + "="*50)
    print("DETECTED TEXT LINES:")
    print("="*50)
    for text in extracted_texts:
        print(f"-> {text}")
        
    print("\n" + "="*50)
    print(" PIPELINE TIMING METRICS:")
    print("="*50)
    print(f"Model Loading: {t_load:.3f} seconds (Ignored in production API)")
    print(f"CRAFT Detect:  {t_detect:.3f} seconds")
    print(f"PARSeq OCR:    {t_recognize:.3f} seconds (Total for {len(extracted_texts)} crops)")
    print(f"Total API Run: {(t_detect + t_recognize):.3f} seconds")
    print("="*50)

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "chianti.jpg"
    main(target_image)