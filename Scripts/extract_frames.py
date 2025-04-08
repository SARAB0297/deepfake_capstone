import os
import cv2
import argparse
from tqdm import tqdm

def extract_frames(video_path, output_folder, frame_rate=5):
    """Extract frames from a video and save them in the output folder."""
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, image = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps is None or fps == 0:
        print(f"Skipping {video_path}: Unable to read FPS.")
        return

    frame_interval = int(fps / frame_rate)

    while success:
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, image)
        success, image = cap.read()
        frame_count += 1

    cap.release()

def process_dataset(dataset_path, output_path, frame_rate=5):
    """Recursively process all video files in the dataset."""
    for root, _, files in os.walk(dataset_path):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)

                # Create a corresponding subdirectory in the output folder
                relative_path = os.path.relpath(root, dataset_path)
                output_folder = os.path.join(output_path, relative_path, file.split('.')[0])
                
                extract_frames(video_path, output_folder, frame_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to FakeAVCeleb dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save extracted frames")
    parser.add_argument("--frame_rate", type=int, default=5, help="Number of frames per second to extract")
    args = parser.parse_args()

    process_dataset(args.dataset, args.output, args.frame_rate)
    print("Frame extraction completed!")
