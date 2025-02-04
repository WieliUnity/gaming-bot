import win32api
import win32con
import time

def right_click_down():
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)

def right_click_up():
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

def move_mouse_relative(dx, dy):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)

def smooth_move(dx, dy, duration, steps):
    """
    Moves the mouse smoothly by dx, dy over the specified duration divided into the given number of steps.
    """
    for i in range(steps):
        move_mouse_relative(dx // steps, dy // steps)
        time.sleep(duration / steps)

# Example usage:
right_click_down()  # Press and hold the right mouse button
smooth_move(100, 0, duration=2, steps=100)  # Move right by 100 units over 2 seconds
right_click_up()  # Release the right mouse button
