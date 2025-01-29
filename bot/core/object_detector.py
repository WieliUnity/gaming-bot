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

        # 2) Forward pass
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
        (Assumes ONNX file was exported with nms=False and single-class.)
        
        For shape (1,5,8400):
          raw[0, :, :] -> shape (5, 8400):
            channel 0 = x_center of all anchors
            channel 1 = y_center of all anchors
            channel 2 = w         of all anchors
            channel 3 = h         of all anchors
            channel 4 = confidence of all anchors
        """
        # The main output is outputs[0], shape = (1, 5, 8400)
        raw = outputs[0]

        # Remove batch dimension => shape (5, 8400)
        raw = np.squeeze(raw, axis=0)
        # debug info
        print("==== _postprocess DEBUG ====")
        print(f"raw.shape = {raw.shape} (should be (5, 8400) for single-class)")

        x_all = raw[0]  # shape (8400,)
        y_all = raw[1]
        w_all = raw[2]
        h_all = raw[3]
        conf_all = raw[4]

        bboxes = []
        confidences = []

        # 1) Parse each anchor
        for i in range(x_all.shape[0]):
            x_center = x_all[i]
            y_center = y_all[i]
            w        = w_all[i]
            h        = h_all[i]
            conf     = conf_all[i]

            # Filter by confidence
            if conf < self.conf_threshold:
                continue

            # If coords look normalized (< ~20?), multiply by 640 if needed
            # But typically these are direct pixel coords if your training used no 'end2end' export.
            # If they're obviously <1, multiply by 640:
            if x_center < 1.5 and w < 1.5:
                x_center *= self.input_size
                y_center *= self.input_size
                w        *= self.input_size
                h        *= self.input_size

            # Convert to [left, top, width, height]
            left = x_center - (w / 2)
            top  = y_center - (h / 2)

            bboxes.append([left, top, w, h])
            confidences.append(float(conf))

        print(f"Total anchors passing conf>{self.conf_threshold}: {len(bboxes)}")

        # 2) NMS
        indices = cv2.dnn.NMSBoxes(bboxes, confidences, self.conf_threshold, self.iou_threshold)
        detections = []

        for idx in indices.flatten():
            x, y, w, h = bboxes[idx]
            conf = confidences[idx]

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
                "label": settings.TARGET_CLASS,
                "confidence": conf,
                "bbox": [left, top, right, bottom]
            })

        print(f"Final detections after NMS: {len(detections)}")
        for det in detections[:5]:
            print("  ", det)

        print("==== End _postprocess DEBUG ====\n")
        return detections

    def process_frame(self, frame, detections, target=None):
        if settings.DEBUG:
            processed_frame = self._draw_detections(frame, detections, target)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(f"{self.debug_dir}/frame_{timestamp}.png", processed_frame)
        return frame

    def _draw_detections(self, frame, detections, target=None):
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
