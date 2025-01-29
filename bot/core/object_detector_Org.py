# bot/core/object_detector.py
import cv2
import os
import numpy as np
from datetime import datetime
from bot.config.settings import settings

class ObjectDetector:
    def __init__(self):
        self.debug_dir = "debug_frames"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Load ONNX model (exported with nms=False)
        self.net = cv2.dnn.readNetFromONNX(settings.MODEL_PATH)
        
        # YOLOv8 default dimensions
        self.input_size = 640
        
        # Confidence threshold from your settings
        self.conf_threshold = settings.CONFIDENCE_THRESHOLD
        # Optional NMS IoU threshold
        self.iou_threshold = 0.45

    def detect(self, frame):
        """Run inference using ONNX model."""
        # 1) Preprocess (direct resize to 640x640 + create blob)
        blob, (ratio_w, ratio_h) = self._preprocess(frame)

        # 2) Run forward pass
        self.net.setInput(blob)
        layer_names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(layer_names)

        # 3) Postprocess
        detections = self._postprocess(outputs, ratio_w, ratio_h)
        return detections

    def _preprocess(self, frame):
        """
        Directly resize the input frame from (H,W) to (640,640).
        We then keep track of the ratio to map boxes back later.
        """
        original_h, original_w = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1/255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False
        )
        ratio_w = original_w / self.input_size
        ratio_h = original_h / self.input_size

        return blob, (ratio_w, ratio_h)

    def _postprocess(self, outputs, ratio_w, ratio_h):
        """
        Convert raw YOLO outputs to detection format and run NMS in code.
        (This assumes the ONNX file was exported with nms=False.)
        """
        # The output shape is typically [1, N, 85] for YOLOv8
        # so let's squeeze out the first dimension, then transpose if needed.
        raw = np.squeeze(outputs[0], axis=0)  # shape: (N, 85) ideally

        # If your model outputs shape is (85, N), do raw = raw.T instead.
        # Make sure you know your model's exact output shape.

        # Separate out the bounding boxes & class predictions
        bboxes = []
        confidences = []
        class_ids = []

        for row in raw:
            x_center, y_center, w, h = row[0], row[1], row[2], row[3]
            class_scores = row[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            # Filter by confidence
            if confidence < self.conf_threshold:
                continue

            # Convert xywh to [left, top, width, height] in 640x640 space
            left = x_center - (w / 2)
            top = y_center - (h / 2)

            bboxes.append([left, top, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

        # Now apply NMS in code
        indices = cv2.dnn.NMSBoxes(
            bboxes,
            confidences,
            self.conf_threshold,
            self.iou_threshold
        )

        detections = []
        for i in indices.flatten():
            x, y, w, h = bboxes[i]
            conf = confidences[i]
            # Scale back to original resolution
            x *= ratio_w
            y *= ratio_h
            w *= ratio_w
            h *= ratio_h

            left   = int(x)
            top    = int(y)
            right  = int(x + w)
            bottom = int(y + h)

            # You can choose the label based on class_ids[i] if you have multiple classes
            detections.append({
                "label": settings.TARGET_CLASS,  # or handle multiple if desired
                "confidence": conf,
                "bbox": [left, top, right, bottom]
            })

        return detections

    def process_frame(self, frame, detections, target=None):
        """Optional: Draw detections for debug purposes."""
        if settings.DEBUG:
            processed_frame = self._draw_detections(frame, detections, target)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(f"{self.debug_dir}/frame_{timestamp}.png", processed_frame)
        return frame

    def _draw_detections(self, frame, detections, target=None):
        """Draw bounding boxes and optionally highlight the locked-on target."""
        overlay_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                overlay_frame, 
                f"{detection['label']} {detection['confidence']:.2f}",
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5,
                (0, 255, 0), 
                2
            )

        if target:
            x1, y1, x2, y2 = target['bbox']
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        return cv2.addWeighted(overlay_frame, 0.7, frame, 0.3, 0)
