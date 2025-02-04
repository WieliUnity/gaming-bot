import cv2
import os
import numpy as np
import time
from datetime import datetime
from bot.config.settings import settings

class ObjectDetector:
    def __init__(self):
        if settings.DEBUG:
            self.debug_dir = settings.DEBUG_DIR
            os.makedirs(self.debug_dir, exist_ok=True)
        
        
        # Load ONNX model (exported with nms=False, multi-class)
        try:
            self.net = cv2.dnn.readNetFromONNX(settings.MODEL_PATH)
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise
        
        # YOLOv8 default dimensions
        self.input_size = settings.INPUT_SIZE
        
        # Confidence threshold from your settings
        self.conf_threshold = settings.CONFIDENCE_THRESHOLD
        # Optional NMS IoU threshold
        self.iou_threshold = settings.IOU_THRESHOLD

    def detect(self, frame): 
        # 1) Preprocess
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
        Resizes the frame directly to 640x640 for YOLO input.
        Computes scaling factors for mapping detections back to the original screen.
        """
        original_h, original_w = frame.shape[:2]  # (1600, 2560)

        # Resize the image to YOLO's input size (without keeping aspect ratio)
        resized = cv2.resize(frame, (self.input_size, self.input_size))

        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1/255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False
        )

        # Compute the direct scaling factors from 640x640 back to 2560x1600
        ratio_w = original_w / self.input_size  # 2560 / 640 = 4.0
        ratio_h = original_h / self.input_size  # 1600 / 640 = 2.5

        #print(f"DEBUG: ratio_w={ratio_w:.3f}, ratio_h={ratio_h:.3f}, original_w={original_w}, original_h={original_h}")

        return blob, (ratio_w, ratio_h)

    def _postprocess(self, outputs, ratio_w, ratio_h):
        """Process YOLOv8 outputs to detections with screen coordinates."""
        # Unpack and verify model outputs
        raw = np.squeeze(outputs[0], 0)
        num_classes = len(settings.CLASS_NAMES)
        
        # Extract prediction components
        xc, yc, w, h = raw[:4]
        class_scores = raw[4:4+num_classes]
        confidences = np.max(class_scores, axis=0)

        # Process each prediction
        detections = []
        for i in np.where(confidences >= self.conf_threshold)[0]:
            # Calculate coordinates
            x = (xc[i] - w[i]/2) * ratio_w
            y = (yc[i] - h[i]/2) * ratio_h
            r = (xc[i] + w[i]/2) * ratio_w
            b = (yc[i] + h[i]/2) * ratio_h
            
            # Clip to screen bounds
            x, y = max(0, int(x)), max(0, int(y))
            r, b = map(lambda v: min(v, settings.MONITOR_REGION["width"]), 
                    [int(r), int(b)])
            
            # Store detection
            class_id = np.argmax(class_scores[:, i])
            detections.append({
                "bbox": [x, y, r, b],
                "confidence": float(confidences[i]),
                "label": settings.CLASS_NAMES[class_id]
            })

        # Apply NMS and format results
        indices = cv2.dnn.NMSBoxes(
            [d["bbox"] for d in detections],
            [d["confidence"] for d in detections],
            self.conf_threshold, self.iou_threshold
        )
        
        # Final filtered results
        final_detections = []
        for idx in (indices.flatten() if isinstance(indices, np.ndarray) else indices):
            det = detections[idx]
            final_detections.append(det)
            #print(f"✅ {det['label']} ({det['confidence']:.2f}) "
            #    f"at {det['bbox']}")

        #print(f"DEBUG: {len(detections)} pre-NMS -> {len(final_detections)} post-NMS")
        return final_detections

    def process_frame(self, frame, detections, target=None):
        """Draw detections and save debug image with matching logs."""
        if settings.DEBUG:
            # Create a unique timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"frame_{timestamp}.png"
            filepath = os.path.join(self.debug_dir, filename)

            # Draw the bounding boxes
            processed_frame = self._draw_detections(frame, detections, target)

            # Save the image
            cv2.imwrite(filepath, processed_frame)

            # ✅ Print log entry with matching filename
            #print(f"📸 Saved debug image: {filename} with {len(detections)} detections.")

        return frame


    def _draw_detections(self, frame, detections, target=None):
        overlay_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Ensure coordinates are correctly drawn
            x1, y1 = max(0, x1), max(0, y1)  # Prevent negative values
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)  # Stay within frame
            
            # Draw green bounding box for all detections
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                overlay_frame,
                f"{detection['label']} {detection['confidence']:.2f}",
                (x1, max(0, y1 - 10)),  # Prevent text from going off-screen
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # Draw red bounding box for the target
        if target:
            x1, y1, x2, y2 = target['bbox']
            x1, y1 = max(0, x1), max(0, y1)  # Prevent negative values
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)  # Stay within frame
            
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                overlay_frame,
                f"{target['label']} {target['confidence']:.2f}",
                (x1, max(0, y1 - 10)),  # Prevent text from going off-screen
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

        return cv2.addWeighted(overlay_frame, 0.7, frame, 0.3, 0)
