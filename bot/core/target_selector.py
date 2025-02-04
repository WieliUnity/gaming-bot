import time
import math
import random
import pyautogui
import cv2
import numpy as np
from bot.config.settings import settings
from bot.core.actions import Actions

class TargetSelector:
    def __init__(self, screen_width, capturer, detection_manager):
        self.screen_center_x = screen_width // 2
        self.screen_width = screen_width
        self.current_target = None
        self.last_target_time = 0
        self.actions = Actions()
        self.tracking = False
        self.capturer = capturer
        self.detection_manager = detection_manager

        # Tracking parameters
        self.rotation_threshold = settings.ROTATION_THRESHOLD
        self.min_target_width   = settings.MIN_TARGET_WIDTH
        self.MAX_TRACKING_TIME  = settings.MAX_TRACKING_TIME
        self.last_rotation      = None

        # Define region for icon search
        self.icon_search_region = (1100, 1170, 100, 130)  # (left, top, width, height)
        self.icon_template = cv2.imread(settings.ICON_TEMPLATE_PATH, cv2.IMREAD_COLOR)

        # Zone definitions
        self.zone_boundaries = {
        'left': (0, settings.ZONE_BOUNDARIES['left'] * screen_width),
        'center': (settings.ZONE_BOUNDARIES['left'] * screen_width, 
                   settings.ZONE_BOUNDARIES['center'] * screen_width),
        'right': (settings.ZONE_BOUNDARIES['center'] * screen_width, screen_width)
    }

    def select_target(self, detections):
        """Pick or maintain a target each tick."""
        if self.tracking:
            return self._maintain_tracking_state()
        return self._find_new_target(detections)

    def _find_new_target(self, detections):
        """Initial detection logic (unchanged)."""
        for priority_class in settings.PRIORITY_TARGETS:
            center_targets = self._filter_detections(detections, priority_class, 'center')
            if center_targets:
                target = self._select_best_target(center_targets)
                return self._start_tracking(target)

            all_targets = [d for d in detections if d['label'] == priority_class]
            if all_targets:
                target = self._select_best_target(all_targets)
                return self._start_tracking(target)
        return None

    def _maintain_tracking_state(self):
        """
        Keep the target in view or initiate interaction if we’re at the correct distance.
        **All interaction here is blocking** (no extra thread).
        """
        if time.time() - self.last_target_time > self.MAX_TRACKING_TIME:
            print("⏰ Tracking timeout")
            self._reset_tracking()
            return None

        time.sleep(0.5)
        new_detections = self.detection_manager.get_latest_predictions()
        if not new_detections:
            return None

        verified_target = self._verify_target_persistence(new_detections)
        if not verified_target:
            print("⚠️ Target lost")
            self._reset_tracking()
            return None

        self._update_current_target(verified_target)

        offset = self._calculate_offset(verified_target)
        width  = self._bbox_width(verified_target['bbox'])

        # 1) If the target is off-center, rotate to recenter.
        if abs(offset) > self.rotation_threshold:
            self._handle_rotation(offset)

        # 2) If the target is still too small, move forward a bit.
        #elif width < self.min_target_width:
        #    self._handle_forward_movement()

        # 3) Otherwise, do the blocking interaction.
        else:
            self._perform_interaction()   # <--- Blocking here
            self._reset_tracking()        # Once done, we can reset or do next steps.

        return self.current_target

    def _perform_interaction(self):

        print("[INTERACTION] Starting blocking interaction...")
        # Press & hold W
        pyautogui.keyDown('w')
        print("[INTERACTION] Walking forward...")   
        start_time = time.time()
        max_walk_time = settings.MAX_WALK_TIME
        counter=1
        found_icon = False
        while time.time() - start_time < max_walk_time:
            if self._icon_appears():
                found_icon = True
                print("[INTERACTION] Icon found, stopping movement.")
                break
            
        
            time.sleep(0.05)

        # Release W
        pyautogui.keyUp('w')
        print("[INTERACTION] Done (icon found?" , found_icon, ")")
        time.sleep(0.5)
        pyautogui.press('f')  # Interact key
        time.sleep(15)  # Wait for interaction to complete

    def _icon_appears(self):
        # Only capture a 1×1 region
        pixel_image = pyautogui.screenshot(region=(1192, 1222, 1, 1))
        
        # Convert to RGB tuple
        r, g, b = pixel_image.getpixel((0, 0))
        
        # Compare to a known color threshold
        if (r, g, b) == settings.ICON_COLOR:
            return True
        return False
    # ------------------------------------
    # Original helper methods below
    # ------------------------------------
    def _calculate_offset(self, target):
        target_center = self._bbox_center_x(target['bbox'])
        return target_center - self.screen_center_x

    def _handle_rotation(self, offset):
        direction = 'right' if offset > 0 else 'left'
        base_distance = abs(offset) * 1.5
        actual_distance = base_distance * random.uniform(0.99, 1.01)
        self.actions.mouse_drag(direction, actual_distance)
        self.last_rotation = {
            'direction': direction,
            'distance': actual_distance,
            'start_pos': self._bbox_center_x(self.current_target['bbox'])
        }
        time.sleep(random.uniform(0.2, 0.4))

    def _handle_forward_movement(self):
        self.actions.press_key('W', 1)
        time.sleep(0.7)

    def _verify_target_persistence(self, new_detections):
        original = self.current_target
        if not original:
            return None
        candidates = [det for det in new_detections if det['label'] == original['label']]
        if not candidates:
            return None

        center_min = self.screen_center_x - (self.screen_width * 0.1)
        center_max = self.screen_center_x + (self.screen_width * 0.1)
        in_center = [det for det in candidates
                     if center_min <= self._bbox_center_x(det['bbox']) <= center_max]
        if in_center:
            return max(in_center, key=lambda d: self._bbox_width(d['bbox']))
        else:
            return min(candidates, key=lambda d: abs(self._bbox_center_x(d['bbox']) - self.screen_center_x))

    def _start_tracking(self, target):
        self.current_target = target
        self.tracking = True
        self.last_target_time = time.time()
        print(f"🎯 New target: {target['label']} at {target['bbox']}")
        
        return target

    def _reset_tracking(self):
        self.tracking = False
        self.current_target = None
        self.last_rotation = None

    def _filter_detections(self, detections, target_class, zone):
        zone_start, zone_end = self.zone_boundaries[zone]
        return [
            d for d in detections
            if d['label'] == target_class
               and self._bbox_center_x(d['bbox']) >= zone_start
               and self._bbox_center_x(d['bbox']) <= zone_end
        ]

    def _select_best_target(self, targets):
        return max(targets, key=lambda d: (
            d['confidence'] * settings.TARGET_SCORE_WEIGHTS[0] +
            self._bbox_width(d['bbox']) * settings.TARGET_SCORE_WEIGHTS[1] +
            (1 / (self._distance_from_center(d['bbox']) + 1)) * settings.TARGET_SCORE_WEIGHTS[2]
        ))

    def _update_current_target(self, target):
        self.current_target = target
        self.last_target_time = time.time()

    def _bbox_center_x(self, bbox):
        return (bbox[0] + bbox[2]) // 2

    def _bbox_width(self, bbox):
        return bbox[2] - bbox[0]

    def _distance_from_center(self, bbox):
        return abs(self._bbox_center_x(bbox) - self.screen_center_x)
