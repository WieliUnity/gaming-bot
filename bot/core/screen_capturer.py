"""
SCREEN CAPTURER MODULE
-----------------------
Handles screen capture using MSS library.
Captures frames in BGR format for OpenCV compatibility.
"""

import cv2
import numpy as np
from mss import mss

class ScreenCapturer:
    """Captures screenshots of a specified screen region."""
    
    def __init__(self, monitor={"top": 0, "left": 0, "width": 1920, "height": 1080}):
        """
        Args:
            monitor (dict): Screen region to capture (default: full HD screen).
        """
        self.sct = mss()
        self.monitor = monitor

    def capture_frame(self) -> np.ndarray:
        """Returns a BGR-formatted numpy array of the captured frame."""
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)