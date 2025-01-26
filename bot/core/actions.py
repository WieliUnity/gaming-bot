"""
ACTIONS MODULE
--------------
Handles mouse/keyboard actions with human-like randomization.
"""

import pyautogui
import random
import time

class Actions:
    def __init__(self):
        # Disable pyautogui's failsafe (use with caution!)
        pyautogui.FAILSAFE = False
        self.screen_width, self.screen_height = pyautogui.size()

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