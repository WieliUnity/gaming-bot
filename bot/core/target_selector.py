import time
from bot.config.settings import settings 

class TargetSelector:
    def __init__(self):
        self.current_target = None
        self.target_lock_duration = 0  # Seconds to keep target if lost
        self.last_target_time = 0

    def select_target(self, detections, frame_width):
        """Select target based on priority list and detection quality."""
        if not detections:
            return None

        # 1) Current target tracking
        if self.current_target:
            current_time = time.time()
            
            # Check for matching target in new detections
            for detection in detections:
                if (detection['label'] == self.current_target['label'] and 
                    self._boxes_overlap(detection['bbox'], self.current_target['bbox'])):
                    self._update_current_target(detection)  # Refresh position
                    return detection

            # Temporary target locking
            if (current_time - self.last_target_time) < self.target_lock_duration:
                return self.current_target

        # 2) Process detections by priority
        for class_name in settings.PRIORITY_TARGETS:
            class_detections = [d for d in detections if d['label'] == class_name]
            
            if not class_detections:
                continue
                
            # Class-specific filtering
            if class_name == "tree":
                valid_targets = [
                    t for t in class_detections
                    if (t['bbox'][2] - t['bbox'][0]) < 0.5 * frame_width
                ]
            else:
                valid_targets = class_detections
                
            if valid_targets:
                best_target = max(
                    valid_targets,
                    key=lambda d: (d['bbox'][2] - d['bbox'][0])  # Widest target
                )
                self._update_current_target(best_target)
                return best_target

        return None

    # Keep the helper methods the same (_update_current_target, _bbox_area, _boxes_overlap)

    # Helper function to maintain original behavior
    def _update_current_target(self, target):
        self.current_target = target
        self.last_target_time = time.time()

    def _bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def _boxes_overlap(self, box1, box2, threshold=0.7):
        # Calculate Intersection over Union (IoU)
        x1_i = max(box1[0], box2[0])
        y1_i = max(box1[1], box2[1])
        x2_i = min(box1[2], box2[2])
        y2_i = min(box1[3], box2[3])

        intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        area1 = self._bbox_area(box1)
        area2 = self._bbox_area(box2)

        iou = intersection / (area1 + area2 - intersection)
        return iou > threshold