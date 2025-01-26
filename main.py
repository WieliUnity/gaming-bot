# main.py
import time
import cv2
from bot.core.screen_capturer import ScreenCapturer
from bot.core.object_detector import ObjectDetector
from bot.core.target_selector import TargetSelector
from bot.config.settings import settings  # <-- Fix this line

def main():
    capturer = ScreenCapturer()
    detector = ObjectDetector()
    selector = TargetSelector()
    
    capturer.start()
    
    try:
        while True:
            frame = capturer.get_frame()
            if frame is not None:
                # Get detections from Roboflow
                detections = detector.detect(frame)
                
                # Select target using your prioritization logic
                target = selector.select_target(detections)
                
                if settings.DEBUG:   # Optional: Highlight the selected target
                    # Process and save frame (no window display)
                    _ = detector.process_frame(frame, detections,target)
            
            time.sleep(0.1)  # Reduce CPU usage
    finally:
        capturer.stop()
        #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()