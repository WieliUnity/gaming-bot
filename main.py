# main.py

import time
import threading
import keyboard  # Install with: pip install keyboard
from bot.core.screen_capturer import ScreenCapturer
from bot.core.object_detector import ObjectDetector
from bot.core.target_selector import TargetSelector
from bot.config.settings import settings

# Global pause flag
paused = False

def toggle_pause():
    """Toggle the pause state when F12 is pressed."""
    global paused   
    while True:
        keyboard.wait("f12")  # Wait for F12 keypress
        paused = not paused
        print("\n[PAUSED]" if paused else "\n[RESUMED]")

def is_paused():
    global paused
    return paused

def main():
    capturer = ScreenCapturer(paused_flag=is_paused)
    detector = ObjectDetector()
    selector = TargetSelector()
    
    capturer.start()

    # Start a separate thread for listening to the pause key
    pause_thread = threading.Thread(target=toggle_pause, daemon=True)
    pause_thread.start()
    print("Pause thread started")

    try:
        while True:
            if paused:
                time.sleep(0.1)  # Avoid high CPU usage while paused
                continue

            frame = capturer.get_frame()
            
            if frame is not None:
                # Get detections from Roboflow
                detections = detector.detect(frame)
                target = selector.select_target(detections, frame.shape[1]) 
                
                if settings.DEBUG:   # Optional: Highlight the selected target
                    _ = detector.process_frame(frame, detections, target)
            
            time.sleep(0.1)  # Reduce CPU usage
    finally:
        capturer.stop()
        #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()