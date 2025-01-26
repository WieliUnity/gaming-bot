# main.py
from bot.core.screen_capturer import ScreenCapturer
from bot.core.object_detector import ObjectDetector
from bot.core.actions import Actions
from bot.core.target_selector import TargetSelector  # <-- Updated import
from bot.config.settings import Settings
import cv2

class GameBot:
    def __init__(self):
        self.settings = Settings()
        self.capturer = ScreenCapturer(self.settings.MONITOR_REGION)
        self.detector = ObjectDetector(
            model_path=self.settings.MODEL_PATH,
            confidence_threshold=self.settings.CONFIDENCE_THRESHOLD
        )
        self.actions = Actions(self.capturer.get_screen_size())
        self.selector = TargetSelector(
            cluster_threshold=100,  # Adjust based on your game's scale
            lock_duration=5
        )

    def draw_debug_overlay(self, frame, boxes):
        """Draw bounding boxes and labels with cluster visualization"""
        # Draw all boxes in gray
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 200, 200), 1)
        
        # Draw current target in green
        if self.selector.current_target:
            x, y, w, h = self.selector.current_target
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Target", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

    def run(self):
        try:
            while True:
                frame = self.capturer.capture_frame()
                boxes, scores, class_ids = self.detector.detect(frame)
                
                if boxes:
                    target_box = self.selector.select_target(boxes)
                    if target_box:
                        x, y, w, h = target_box
                        center_x = x + w//2
                        center_y = y + h//2
                        self.actions.human_click(center_x, center_y)
                
                # Debug visualization
                debug_frame = self.draw_debug_overlay(frame.copy(), boxes)
                cv2.imshow("Bot Debug View", debug_frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            print("\nBot stopped by user")

if __name__ == "__main__":
    bot = GameBot()
    bot.run()