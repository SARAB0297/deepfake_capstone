import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths
input_root = "dataset/FakeAVCeleb_Frames"
output_root = "dataset/FakeAVCeleb_Preprocessed"
target_size = (224, 224)

# Toggle standardization
APPLY_STANDARDIZATION = True

# ImageNet mean & std (optional)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def preprocess_and_save(image_path, output_path):
    # Read color image (ensure no grayscale conversion)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Warning: Failed to read {image_path}. Skipping...")
        return

    # Resize using Lanczos interpolation for quality preservation
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Convert to float32 for precise operations
    resized = resized.astype(np.float32)

    # Normalize using ImageNet mean/std (or per-image standardization)
    if APPLY_STANDARDIZATION:
        mean = np.mean(resized, axis=(0, 1), keepdims=True)
        std = np.std(resized, axis=(0, 1), keepdims=True) + 1e-8
        standardized = (resized - mean) / std
    else:
        standardized = (resized / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

    # Scale back to [0, 255] and save as high-quality JPG
    final_img = np.clip(standardized * 255, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final_img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def process_all():
    for root, _, files in os.walk(input_root):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.lower().endswith((".jpg", ".png")):
                in_path = os.path.join(root, file)
                rel_path = os.path.relpath(in_path, input_root)
                out_path = os.path.join(output_root, rel_path)
                preprocess_and_save(in_path, out_path)


if __name__ == "__main__":
    process_all()
    print("✅ Preprocessing complete. Saved to:", output_root)
