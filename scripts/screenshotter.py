# scripts/dataset_capture.py
import sys
import os
import cv2
import time
import keyboard  # Install with `pip install keyboard`
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot.core.screen_capturer import ScreenCapturer
# Add project root to Python path

# Set the output folder
OUTPUT_DIR = r"C:\Python Projects\gaming-bot\train_data"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the screen capture region (modify if needed)
REGION = {"top": 0, "left": 0, "width": 2560, "height": 1600}

def capture_training_data():
    """
    Capture screenshots only when F12 is pressed.
    """
    capturer = ScreenCapturer(monitor=REGION)
    capturer.start()

    print("[INFO] Press F12 to capture a screenshot. Press Ctrl+C to exit.")

    try:
        while True:
            keyboard.wait("F12")  # Wait for F12 key press
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame = capturer.get_frame()  # Capture the frame
            
            if frame is not None:
                filename = os.path.join(OUTPUT_DIR, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[CAPTURED] Screenshot saved: {filename}")

            time.sleep(0.2)  # Small delay to prevent multiple captures from one keypress
    except KeyboardInterrupt:
        print("\n[INFO] Screenshot capture stopped.")

if __name__ == "__main__":
    capture_training_data()
