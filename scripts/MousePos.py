import tkinter as tk
from pynput import mouse, keyboard
from pynput.keyboard import Key, Listener as KeyboardListener
import pyautogui
import threading

# Global variables
tracking_active = True
root = None
label = None

def toggle_tracking():
    """
    Toggles the mouse tracking on and off.
    """
    global tracking_active
    tracking_active = not tracking_active
    if tracking_active:
        label.config(text="Tracking ON")
    else:
        label.config(text="Tracking PAUSED")

def update_mouse_position(x, y):
    """
    Updates the label with the current mouse coordinates and pixel color.
    """
    if tracking_active:
        # Attempt to read the pixel color at (x, y)
        color_text = get_pixel_color_text(x, y)
        label.config(text=f"Mouse Position: ({x}, {y})  {color_text}")
        root.update_idletasks()

def get_pixel_color_text(x, y):
    """
    Captures a 1x1 screenshot at (x,y) and returns an RGB color string.
    If out of bounds, returns an empty string.
    """
    screen_width, screen_height = pyautogui.size()
    # Ensure x,y is within the screen bounds to avoid errors
    if not (0 <= x < screen_width and 0 <= y < screen_height):
        return "[Out of bounds]"
    
    # Capture a 1x1 region
    pixel_image = pyautogui.screenshot(region=(x, y, 1, 1))
    # pixel_image is a PIL Image in RGB mode
    r, g, b = pixel_image.getpixel((0, 0))
    
    return f"Color: (R={r}, G={g}, B={b})"

def start_gui():
    """
    Starts the Tkinter GUI with the label.
    """
    global root, label
    root = tk.Tk()
    root.title("Mouse Tracker")
    root.geometry("600x60")
    root.resizable(False, False)
    label = tk.Label(root, text="Tracking ON", font=("Helvetica", 12))
    label.pack(fill=tk.BOTH, expand=True)
    root.attributes('-topmost', True)  # Keep the window on top
    return root

def keyboard_listener():
    """
    Listens for the F12 key to toggle tracking.
    """
    from pynput.keyboard import Key, Listener as KeyboardListener
    def on_press(key):
        try:
            if key == Key.f12:
                toggle_tracking()
        except Exception as e:
            print(f"Error: {e}")

    with KeyboardListener(on_press=on_press) as listener:
        listener.join()

def mouse_listener():
    """
    Listens for mouse movement events.
    """
    def on_move(x, y):
        update_mouse_position(x, y)

    with mouse.Listener(on_move=on_move) as listener:
        listener.join()

if __name__ == "__main__":
    # Start the Tkinter GUI
    root = start_gui()

    # Start mouse listener in a separate thread
    mouse_thread = threading.Thread(target=mouse_listener, daemon=True)
    mouse_thread.start()

    # Start keyboard listener in a separate thread
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()

    # Run the Tkinter main loop
    root.mainloop()
