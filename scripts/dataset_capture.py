# scripts/dataset_capture.py
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Rest of your imports
import cv2
import time
from datetime import datetime
from bot.core.screen_capturer import ScreenCapturer

def capture_training_data(
    output_dir: str = "train_data",
    interval: int = 2,          # Seconds between captures
    max_captures: int = 1000,   # Max images to collect
    region: dict = {"top": 100, "left": 0, "width": 1920, "height": 980}  # Exclude UI
):
    os.makedirs(output_dir, exist_ok=True)
    capturer = ScreenCapturer(monitor=region)
    
    print(f"Capturing {max_captures} screenshots to {output_dir}...")
    for i in range(max_captures):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame = capturer.capture_frame()
        cv2.imwrite(f"{output_dir}/capture_{timestamp}_{i}.jpg", frame)
        time.sleep(interval)
        print(f"Captured image {i+1}/{max_captures}")

if __name__ == "__main__":
    capture_training_data()