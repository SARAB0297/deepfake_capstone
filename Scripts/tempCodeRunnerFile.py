 # Resize using Lanczos interpolation for quality preservation
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)