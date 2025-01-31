"""
SCREEN CAPTURER MODULE
-----------------------
Handles screen capture using DXCAM library.
Captures frames in BGR format for OpenCV compatibility.
"""
import time
import dxcam
import threading
import numpy as np
from bot.config.settings import settings  # <-- Import settings

class ScreenCapturer:
    def __init__(self, monitor=None, paused_flag=None):
        self.monitor = monitor or settings.MONITOR_REGION
        self.paused_flag = paused_flag  # Add paused_flag parameter
        # Convert dict -> (left, top, right, bottom)
        self.dxcam_region = (
            self.monitor["left"],
            self.monitor["top"],
            self.monitor["left"] + self.monitor["width"],   # right
            self.monitor["top"] + self.monitor["height"],   # bottom
        )

        self.latest_frame = None
        self.running = False
        self.thread = None
        self.camera = dxcam.create(output_color="BGR")
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            if self.paused_flag and self.paused_flag():
                time.sleep(0.1)  # Avoid high CPU usage while paused
                continue

            frame_bgr = self.camera.grab(region=self.dxcam_region)
            if frame_bgr is not None:
                frame_np = np.array(frame_bgr)
                with self.lock:
                    self.latest_frame = frame_np
                    #if settings.DEBUG:
                        #print("Captured frame shape:", frame_np.shape)
                #self.latest_frame = np.array(frame_bgr)

    def get_frame(self):
        # Return a copy so we don't accidentally modify the live frame
         with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
        #return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
