# main.py

import time
import threading
import keyboard  # pip install keyboard

from bot.core.screen_capturer import ScreenCapturer
# from bot.core.object_detector import ObjectDetector  # <-- No longer used in main
from bot.core.target_selector import TargetSelector
from bot.config.settings import settings
from bot.core.detection_manager import DetectionManager

# Global pause flag
paused = False

def toggle_pause():
    """Toggle the pause state when F12 is pressed."""
    global paused
    while True:
        keyboard.wait("f12")
        paused = not paused
        print("\n[PAUSED]" if paused else "\n[RESUMED]")

def is_paused():
    global paused
    return paused

def main():
    # 1) Create screen capturer
    capturer = ScreenCapturer(paused_flag=is_paused)

    # 2) Create detection manager (4 threads, for example)
    detection_manager = DetectionManager(
        capturer=capturer,
        paused_flag=is_paused,
        num_workers=settings.MAX_DETECTION_WORKERS
    )

    # 3) Start capturing and detection threads
    capturer.start()
    detection_manager.start()

    # 4) Create TargetSelector
    screen_width = capturer.dxcam_region[2] - capturer.dxcam_region[0]
    selector = TargetSelector(
        screen_width=screen_width,
        capturer=capturer,       # We still pass it in, as it may be used inside TargetSelector
        detection_manager=detection_manager
    )

    # 5) Thread to handle F12 pause togglingf
    pause_thread = threading.Thread(target=toggle_pause, daemon=True)
    pause_thread.start()
    print("Pause thread started")

    try:
        while True:
            if paused:
                time.sleep(0.1)
                continue
            
            # The biggest difference: we do NOT call "capturer.wframe()" -> "detector.detect(...)"
            # Instead, we ask detection_manager for the *latest* predictiofns.
            
            # [Optional] Sleep ~0.7s so that the predictions are sure to be up-to-date
            time.sleep(0.5)

            # Retrieve the latest predictions from the detection threads
            detections = detection_manager.get_latest_predictions()

            # Then pass them into the TargetSelector
            target = selector.select_target(detections)

            # If in DEBUG mode, we can visualize or do something.
            # But we no longer need to call process_frame or anything,
            # unless you want to show bounding boxes on your own.
            if settings.DEBUG:
                pass  # e.g., you could call your debug drawing code here

            time.sleep(0.1)  # Slow down main loop slightly

    finally:
        # Cleanup
        capturer.stop()
        detection_manager.stop()
        print("Shutting down cleanly.")

if __name__ == "__main__":
    main()
