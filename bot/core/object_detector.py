"""
OBJECT DETECTOR MODULE
----------------------
Handles YOLOv8 model loading and inference using ONNX.
"""

import cv2
import onnxruntime as ort
import numpy as np

class ObjectDetector:
    """Detects objects in frames using a pre-trained ONNX model."""
    
    def __init__(self, model_path: str = "bot/models/tree_model.onnx"):
        """
        Args:
            model_path (str): Path to ONNX model file.
        """
        self.session = ort.InferenceSession(model_path)
        self.input_shape = self.session.get_inputs()[0].shape[2:]  # (height, width)

    def detect(self, frame: np.ndarray) -> tuple:
        """Detects objects in a frame.
        
        Returns:
            tuple: (boxes, scores, class_ids)
        """
        blob = self._preprocess(frame)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        return self._postprocess(outputs, frame.shape)

    # bot/core/object_detector.py
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resizes and normalizes frame for ONNX model input."""
        resized = cv2.resize(frame, (640, 640))  # Match Roboflow’s resize
        blob = cv2.dnn.blobFromImage(resized, 1/255.0, swapRB=True)
        return blob.astype(np.float32)
