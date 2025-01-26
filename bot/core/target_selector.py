# bot/core/target_selector.py
import time

class TargetSelector:
    def __init__(self):
        self.current_target = None
        self.target_lock_duration = 5  # Seconds to keep target if lost
        self.last_target_time = 0

    def select_target(self, detections):
        if not detections:
            return None

        # If we have a current target that's still visible, keep it
        if self.current_target:
            for detection in detections:
                if detection['label'] == self.current_target['label'] and \
                   self._boxes_overlap(detection['bbox'], self.current_target['bbox']):
                    return self.current_target

            # If target lost but within lock duration, keep it
            if (time.time() - self.last_target_time) < self.target_lock_duration:
                return self.current_target

        # Select new target based on largest bounding box area
        largest = max(detections, key=lambda x: self._bbox_area(x['bbox']))
        self.current_target = largest
        self.last_target_time = time.time()
        return largest

    def _bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def _boxes_overlap(self, box1, box2, threshold=0.7):
        # Calculate intersection over union (IoU)
        x1_i = max(box1[0], box2[0])
        y1_i = max(box1[1], box2[1])
        x2_i = min(box1[2], box2[2])
        y2_i = min(box1[3], box2[3])
        
        intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        area1 = self._bbox_area(box1)
        area2 = self._bbox_area(box2)
        
        iou = intersection / (area1 + area2 - intersection)
        return iou > threshold