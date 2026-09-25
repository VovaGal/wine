import sys
import torch
from torchvision import transforms
from PIL import Image

# Import your expansion function directly from your training script
from ocr_10_epoch import expand_parseq

def test_image(image_path, checkpoint_path="weights/parseq_wine_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Testing on device: {device}")

    # 1. Load the checkpoint dictionary
    print(f"[*] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 2. Initialize the base model (pretrained=False because we overwrite it anyway)
    print("[*] Initializing PARSeq architecture...")
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=False).to(device)

    # 3. Expand the architecture using the exact parameters saved in your checkpoint
    expand_parseq(
        model, 
        checkpoint["charset"], 
        checkpoint["max_label_length"], 
        device
    )

    # 4. Load your fine-tuned weights
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 5. Apply the exact same visual transformations used during training
    img_transform = transforms.Compose([
        transforms.Resize((32, 128), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 6. Load and prepare the image
    img = Image.open(image_path).convert("RGB")
    img_tensor = img_transform(img).unsqueeze(0).to(device)

    # 7. Run inference
    print(f"[*] Running inference on {image_path}...\n")
    with torch.inference_mode():
        # Without target labels provided, PARSeq automatically generates text autoregressively
        logits = model(img_tensor)
        
        # Convert logits to probabilities and decode to strings
        pred = logits.softmax(-1)
        label, confidence = model.tokenizer.decode(pred)

    print("="*50)
    print(f" PREDICTED TEXT: {label[0]}")
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_inference.py <path_to_cropped_image.jpg>")
    else:
        test_image(sys.argv[1])