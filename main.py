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
                
                # Optional: Highlight the selected target
                if target:
                    x1, y1, x2, y2 = target['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Show overlay with all detections + target highlight
                processed_frame = detector.process_frame(frame, detections)
                if settings.SHOW_OVERLAY:
                    cv2.imshow('Bot Overlay', processed_frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
            
            time.sleep(0.1)  # Reduce CPU usage
    finally:
        capturer.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()