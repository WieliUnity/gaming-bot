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
        """
        # The output shape is (1, 84, 8400) for YOLOv8
        raw = np.squeeze(outputs[0], axis=0)  # Removes batch dimension: (84, 8400)
        raw = raw.T  # Transpose to get (8400, 84) where each row is a detection

        bboxes = []
        confidences = []
        class_ids = []

        for row in raw:
            # Denormalize coordinates: multiply by input_size to get pixel values
            x_center = row[0] * self.input_size
            y_center = row[1] * self.input_size
            w = row[2] * self.input_size
            h = row[3] * self.input_size

            class_scores = row[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence < self.conf_threshold:
                continue

            # Convert xywh to [left, top, width, height] in 640x640 space
            left = x_center - (w / 2)
            top = y_center - (h / 2)

            bboxes.append([left, top, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes,
            confidences,
            self.conf_threshold,
            self.iou_threshold
        )

        detections = []
        for i in indices.flatten():
            x, y, w, h = bboxes[i]
            # Scale back to original resolution
            x *= ratio_w
            y *= ratio_h
            w *= ratio_w
            h *= ratio_h

            left   = int(x)
            top    = int(y)
            right  = int(x + w)
            bottom = int(y + h)

            detections.append({
                "label": settings.CLASS_NAMES[class_ids[i]],
                "confidence": confidences[i],
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
