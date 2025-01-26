# bot/core/object_detector.py
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from bot.config import settings

class ObjectDetector:
    def __init__(self):
        self.overlay_enabled = True
        # Initialize Roboflow client
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=settings.ROBOFLOW_API_KEY
        )
        self.model_id = "gaming-bot/2"

    def detect(self, frame):
        """Run inference using Roboflow SDK"""
        # Convert BGR to RGB (Roboflow expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.client.infer(rgb_frame, model_id=self.model_id)
        return self._parse_results(results)

    def _parse_results(self, results):
        """Convert Roboflow predictions to our detection format"""
        detections = []
        for prediction in results.get("predictions", []):
            # Convert center-based coordinates to corner-based
            x = prediction["x"]
            y = prediction["y"]
            width = prediction["width"]
            height = prediction["height"]
            
            detections.append({
                "label": prediction["class"],
                "confidence": prediction["confidence"],
                "bbox": [
                    int(x - width/2),  # x1
                    int(y - height/2), # y1
                    int(x + width/2),  # x2
                    int(y + height/2)  # y2
                ]
            })
        return detections

    def process_frame(self, frame, detections):
        """Add overlay to frame (unchanged from your original)"""
        if settings.SHOW_OVERLAY and self.overlay_enabled:
            return self._draw_detections(frame, detections)
        return frame

    def _draw_detections(self, frame, detections):
        """Draw detection boxes (unchanged from your original)"""
        overlay_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay_frame, 
                       f"{detection['label']} {detection['confidence']:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0), 2)
        
        return cv2.addWeighted(overlay_frame, 0.7, frame, 0.3, 0)