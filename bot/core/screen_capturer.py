"""
SCREEN CAPTURER MODULE
-----------------------
Handles screen capture using MSS library.
Captures frames in BGR format for OpenCV compatibility.
"""

import mss
import threading
import numpy as np
from bot.config.settings import settings  # <-- Import settings

class ScreenCapturer:
    def __init__(self):
        self.monitor = settings.MONITOR_REGION  # <-- Use settings
        self.latest_frame = None
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        with mss.mss() as sct:
            while self.running:
                raw = sct.grab(self.monitor)  # Now uses the dictionary
                self.latest_frame = np.array(raw)

    def get_frame(self):
        return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()