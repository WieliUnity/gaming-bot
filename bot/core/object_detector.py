# bot/core/object_detector.py
import cv2
import os
import numpy as np
from inference_sdk import InferenceHTTPClient
from bot.config.settings import settings  # <-- Import the instance
from datetime import datetime

class ObjectDetector:
    def __init__(self):
        self.overlay_enabled = True
        self.debug_dir = "debug_frames"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Load ONNX model
        self.net = cv2.dnn.readNetFromONNX(settings.MODEL_PATH)
        self.input_size = 640  # YOLOv8 default
        
        # Keep original color conversion flag
        self.color_conversion = cv2.COLOR_BGR2RGB 

    def detect(self, frame):
        """Run inference using ONNX model"""
        # Preprocess
        blob, ratio = self._preprocess(frame)
        
        # Inference
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        
        # Post-process
        detections = self._postprocess(outputs, ratio)
        return detections

    def _preprocess(self, frame):
        """Resize and normalize image for YOLOv8"""
        # Keep aspect ratio
        h, w = frame.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        blob = cv2.dnn.blobFromImage(
            resized, 
            1/255.0, 
            (self.input_size, self.input_size), 
            swapRB=True,  # Maintain RGB order
            crop=False
        )
        
        return blob, scale

    def _postprocess(self, outputs, ratio):
        """Convert raw outputs to detection format"""
        detections = []
        outputs = np.squeeze(outputs[0]).T
        
        # Filter by confidence
        conf_threshold = settings.CONFIDENCE_THRESHOLD
        scores = outputs[:, 4:]
        max_scores = np.max(scores, axis=1)
        mask = max_scores > conf_threshold
        outputs = outputs[mask]
        
        for output in outputs:
            # Extract class with highest score
            class_id = np.argmax(output[4:])
            confidence = output[4 + class_id]
            
            # Get original coordinates
            x, y, w, h = output[0], output[1], output[2], output[3]
            
            # Scale to original image
            left = int((x - w/2) / ratio)
            top = int((y - h/2) / ratio)
            right = int((x + w/2) / ratio)
            bottom = int((y + h/2) / ratio)
            
            detections.append({
                "label": settings.TARGET_CLASS,  # From your settings
                "confidence": float(confidence),
                "bbox": [left, top, right, bottom]
            })
            
        return detections
    def process_frame(self, frame, detections,target=None):
        
        if settings.DEBUG:
            processed_frame = self._draw_detections(frame, detections,target)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(f"{self.debug_dir}/frame_{timestamp}.png", processed_frame)
        return frame
    
    def _draw_detections(self, frame, detections, target=None):
        """Draw detection boxes (unchanged from your original)"""
        overlay_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay_frame, 
                       f"{detection['label']} {detection['confidence']:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0), 2)
    # Draw target in red (on top of detections)
        if target:
            x1, y1, x2, y2 = target['bbox']
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        return cv2.addWeighted(overlay_frame, 0.7, frame, 0.3, 0)