import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.filters import gabor_kernel
from scipy.ndimage import convolve
from datetime import datetime

# ---------- Config ----------
INPUT_ROOT = r"D:/Deepfake_/dataset/FakeAVCeleb_Preprocessed"
OUTPUT_CSV = r"D:/Deepfake_/dataset/FakeAVCeleb_LogGabor_Features.csv"
LOG_FILE = r"D:/Deepfake_/dataset/skipped_files.log"
OUTPUT_IMAGE_ROOT = r"D:/Deepfake_/dataset/FakeAVCeleb_LogGabor_Images"

FREQUENCIES = [0.1, 0.2, 0.3]
ORIENTATIONS = [0, np.pi/4, np.pi/2, 3*np.pi/4]
KERNEL_SIZE = 21
MAX_FEATURES = 108


# ---------- Feature Extraction ----------
def extract_log_gabor_features_and_visual(image, save_path):
    features = []
    vis_output = np.zeros_like(image)

    for freq in FREQUENCIES:
        for theta in ORIENTATIONS:
            kernel = np.real(gabor_kernel(freq, theta=theta))
            for c in range(3):  # RGB
                response = convolve(image[:, :, c], kernel, mode='reflect')

                # Add to visual output (normalize per-channel)
                norm_resp = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
                vis_output[:, :, c] += norm_resp.astype(np.uint8) // (len(FREQUENCIES) * len(ORIENTATIONS))

                # Append features
                features.extend([
                    np.mean(response),
                    np.std(response),
                    np.sum(response ** 2)
                ])

    # Save transformed visualization
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis_output, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return np.array(features)


# ---------- Main Processing ----------
def extract_features_streaming():
    skipped_files = []

    headers = ['frame_name', 'label'] + ['feat_{}'.format(i) for i in range(MAX_FEATURES)]

    with open(OUTPUT_CSV, 'w', encoding='utf-8') as f:
        f.write(','.join(headers) + '\n')

    print(f"Starting feature extraction from: {INPUT_ROOT}")
    start_time = datetime.now()

    for root, _, files in os.walk(INPUT_ROOT):
        for file in tqdm(files, desc=f"Processing {root}"):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            full_path = os.path.join(root, file)

            try:
                img = cv2.imread(full_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Unreadable image")

                img = img.astype(np.float32) / 255.0

                # Save path for transformed frame
                rel_path = os.path.relpath(full_path, INPUT_ROOT)
                transformed_frame_path = os.path.join(OUTPUT_IMAGE_ROOT, rel_path)

                features = extract_log_gabor_features_and_visual(img, transformed_frame_path)

                if len(features) > MAX_FEATURES:
                    features = features[:MAX_FEATURES]
                else:
                    features = np.pad(features, (0, MAX_FEATURES - len(features)))

                label = rel_path.split(os.sep)[0]
                frame_name = os.path.basename(full_path)

                row = [frame_name, label] + features.tolist()

                with open(OUTPUT_CSV, 'a', encoding='utf-8') as f:
                    f.write(','.join(map(str, row)) + '\n')

            except Exception as e:
                skipped_files.append(full_path)
                print(f"Skipped {full_path}: {e}")
                continue

    if skipped_files:
        with open(LOG_FILE, 'w') as log:
            log.write('\n'.join(skipped_files))
        print(f"\n Skipped files logged to: {LOG_FILE}")

    print(f"\n Feature extraction complete.")
    print(f" Features saved to: {OUTPUT_CSV}")
    print(f" Transformed frames saved to: {OUTPUT_IMAGE_ROOT}")
    print(f" Total time: {datetime.now() - start_time}")


# ---------- Run ----------
if __name__ == "__main__":
    extract_features_streaming()
