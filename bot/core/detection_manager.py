import threading
import time
from collections import deque
import copy

from bot.core.object_detector import ObjectDetector
from bot.config.settings import settings

class DetectionManager:
    """
    Spawns multiple threads for object detection. Each thread:
      - Continuously grabs the LATEST frame from the ScreenCapturer.
      - Runs detection with its own ObjectDetector instance.
      - Publishes the results to a shared "latest predictions" buffer.

    The main thread can then retrieve the most recent predictions
    via get_latest_predictions().
    """
    def __init__(self, capturer, paused_flag=None, num_workers=None):
        self.capturer = capturer
        self.paused_flag = paused_flag
        self.num_workers = num_workers if num_workers is not None else settings.MAX_DETECTION_WORKERS

        # Each worker has its own ObjectDetector
        self.detectors = [ObjectDetector() for _ in range(self.num_workers)]

        # We store the most recent predictions (and the timestamp/frame_id).
        # Could also store them in a queue, but let's keep only the latest.
        self.lock = threading.Lock()
        self.latest_predictions = []
        self.latest_frame_id = 0
        self.last_processed_frame_id = [-1] * self.num_workers  # Track each worker's last processed frame

        # Worker threads
        self.workers = []
        self.running = False

        # Optional: We can keep a "frame counter" if we want to guarantee
        # that each new frame has a unique ID. But for simplicity,
        # we can rely on the reference or a time-based approach.
        self._frame_counter = 0

    def start(self):
        self.running = True
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            self.workers.append(t)
            t.start()

    def stop(self):
        self.running = False
        for t in self.workers:
            t.join()

    def _worker_loop(self, worker_id):
        """Continuously fetch frames + run detection."""
        detector = self.detectors[worker_id]
        while self.running:
            if self.paused_flag and self.paused_flag():
                time.sleep(0.1)
                continue

            # Get the latest frame from the capturer
            frame = self.capturer.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # We can do a quick check to avoid re-detecting the exact same
            # frame if nothing has changed:
            current_id = id(frame)  # or you could do a custom counter
            if current_id == self.last_processed_frame_id[worker_id]:
                time.sleep(0.01)
                continue

            # This is a new frame => run detection
            self.last_processed_frame_id[worker_id] = current_id
            detections = detector.detect(frame)

            # Update the global "latest_predictions" buffer
            with self.lock:
                # We simply store the newest detections in one shared list.
                # If multiple workers finish around the same time, we keep
                # whichever is last. You could store all but we’ll keep it simple.
                self.latest_predictions = detections

            # Possibly add small sleep to reduce CPU usage
            time.sleep(0.01)

    def get_latest_predictions(self):
        """Return the most recent detections in a threadsafe manner."""
        with self.lock:
            return copy.deepcopy(self.latest_predictions)
