# mask.py
import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2

# Import paths from config
try:
    from config import INPUTS_FOLDER, MASKS_FOLDER, MODEL_PATH, UNET_PY_PATH
    print("[INFO] Configuration loaded successfully.")
except ImportError as e:
    raise RuntimeError("Could not load config. Make sure config.py is in PYTHONPATH") from e

# Add model path to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import U-Net model and get_transform
from src.models.unet import UNet
from src.utils import get_transform

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# List of class IDs that represent clothing
CLOTHING_CLASSES = {
    4,  # Upper-clothes
    5,  # Skirt
    6,  # Pants
    7,  # Dress
    8,  # Belt
    9,  # Right-shoe
    10, # Left-shoe
    16, # Bag
    17, # Scarf
}

# Define DPI for mm-to-pixel conversion
DPI = 170  # Standard high-res value
MM_TO_PIXEL = DPI / 25.4  # 1 inch = 25.4 mm


def segment_image(image, model):
    original_width, original_height = image.size
    transform = get_transform(model.mean, model.std)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        mask = model(input_tensor)
    mask = torch.argmax(mask.squeeze(), dim=0).cpu().numpy()

    # Step 1: Create binary mask of only clothing parts
    binary_mask = np.zeros_like(mask, dtype=np.uint8)
    for cls_id in CLOTHING_CLASSES:
        binary_mask[mask == cls_id] = 255  # White for clothing

    # Step 2: Dilate the mask to expand clothing regions (~2mm)
    pixel_padding = int(1 * MM_TO_PIXEL)  # 2mm padding
    kernel_size = max(1, int(pixel_padding * 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Step 3: Smooth the edges using Gaussian blur and thresholding
    smoothed_mask = cv2.GaussianBlur(dilated_mask, (5, 5), 0)
    _, final_binary_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)

    # Convert to RGB and resize/center on original image
    binary_mask_rgb = np.stack([final_binary_mask]*3, axis=-1)
    mask_image = Image.fromarray(binary_mask_rgb)
    mask_aspect_ratio = mask_image.width / mask_image.height

    new_height = original_height
    new_width = int(new_height * mask_aspect_ratio)
    mask_image = mask_image.resize((new_width, new_height), Image.NEAREST)

    final_mask = Image.new("RGB", (original_width, original_height))
    offset = ((original_width - new_width) // 2, 0)
    final_mask.paste(mask_image, offset)

    return final_mask


def main(input_path, output_path):
    print("[INFO] Loading U-Net model...")
    unet = UNet()
    unet.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    unet.to(device)
    unet.eval()

    try:
        print(f"[INFO] Processing image: {input_path}")
        image = Image.open(input_path).convert("RGB")
        result = segment_image(image, unet)
        result.save(output_path)
        print(f"[INFO] Saved mask to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to process image: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python mask.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    main(input_path, output_path)