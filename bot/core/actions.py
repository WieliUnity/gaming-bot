"""
ACTIONS MODULE
--------------
Handles mouse/keyboard actions with human-like randomization.
"""

import pyautogui
import random
import time
import win32api
import win32con

class Actions:
    def __init__(self):
        # Disable pyautogui's failsafe (use with caution!)
        pyautogui.FAILSAFE = False
        self.screen_width, self.screen_height = pyautogui.size()
        self.MOUSE_SENSITIVITY = 1.0  # Pixels per ms of movement (needs calibration)

    def human_move(self, x: int, y: int):
        """Moves mouse to (x,y) with human-like jitter/delays."""
        # Add slight offset to target
        x += random.randint(-5, 5)
        y += random.randint(-5, 5)
        
        # Non-linear movement duration
        duration = random.uniform(0.3, 0.7)
        pyautogui.moveTo(x, y, duration=duration)

    def human_click(self, x: int, y: int, button: str = 'left'):
        """Clicks at (x,y) with randomized delays."""
        self.human_move(x, y)
        time.sleep(random.uniform(0.1, 0.3))  # Pretend to "aim"
        pyautogui.click(button=button)
        time.sleep(random.uniform(0.2, 0.5))  # Post-click delay

    def press_key(self, key: str, repeats: int = 1):
        """Presses a key with human-like timing."""
        for _ in range(repeats):
            pyautogui.keyDown(key)
            time.sleep(random.uniform(0.05, 0.1))
            pyautogui.keyUp(key)
            time.sleep(random.uniform(0.1, 0.3))
    def rotate(self, direction, duration):
        """Perform human-like rotation"""
        key = 'D' if direction == 'right' else 'A'
        self._human_key_press(key, duration)

    def _human_key_press(self, key, duration):
        """Press key with human-like variance"""
        actual_duration = duration * random.uniform(0.8, 1.2)
        pyautogui.keyDown(key)
        time.sleep(actual_duration)
        pyautogui.keyUp(key)
        time.sleep(random.uniform(0.1, 0.3))
        
    def right_click_down(self):
        """Simulate pressing the right mouse button down."""
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)

    def right_click_up(self):
        """Simulate releasing the right mouse button."""
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

    def move_mouse_relative(self, dx, dy):
        """
        Move the mouse by a relative amount.
        dx and dy should be integers.
        """
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

    def ease_out_quad(self, t):
        """
        Ease-out quadratic tween function.
        t should be in the range [0, 1].
        """
        return 1 - (1 - t) * (1 - t)

    def smooth_move_relative(self, dx, dy, duration, steps, easing_func=None):
        """
        Move the mouse smoothly by dx, dy over the specified duration and steps.
        An easing function can be provided for non-linear movement.
        """
        if easing_func is None:
            easing_func = lambda t: t

        cumulative_x = 0.0
        cumulative_y = 0.0
        for step in range(1, steps + 1):
            t = step / steps
            eased_t = easing_func(t)
            target_x = dx * eased_t
            target_y = dy * eased_t
            move_dx = target_x - cumulative_x
            move_dy = target_y - cumulative_y
            self.move_mouse_relative(move_dx, move_dy)
            cumulative_x = target_x
            cumulative_y = target_y
            time.sleep(duration / steps)

    def mouse_drag(self, direction, distance):
        """
        Perform a human-like right-click drag rotation.
        This method simulates pressing and holding the right mouse button,
        then moving the mouse smoothly in the specified direction ('right' or 'left')
        by the given distance in pixels, and finally releasing the button.
        """

        print(f"Performing a {direction} drag by {distance} pixels...")
        # Press and hold the right mouse button
        self.right_click_down()
        time.sleep(0.15)  # Allow time for click registration

        # Determine horizontal movement (negative for left drag)
        dx = distance if direction.lower() == 'right' else -distance
   
        dy = 0  # No vertical movement

        # Define smooth movement parameters
        num_steps = 85# random.randint(80, 85)
        duration = 0.30#random.uniform(0.30 , 0.35)

        # Perform smooth movement with an ease-out quadratic effect
        self.smooth_move_relative(dx, dy, duration, num_steps, easing_func=self.ease_out_quad)

        # Release the right mouse button
        self.right_click_up()
        time.sleep(random.uniform(0.2, 0.4))