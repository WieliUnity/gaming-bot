class Settings:
    #showing object detection overlay
    DEBUG: bool = True
    DEBUG_DIR: str = "debug_frames"  # Add this line
    
    
    # Screen capture
    MONITOR_REGION = {
        "top": 0,
        "left": 0,
        "width": 1920,
        "height": 1080
    }
    
    # Object detection
    CONFIDENCE_THRESHOLD = 0.5
    MODEL_PATH = "bot/models/tree_model_no1.onnx"
    TARGET_CLASS = "tree"  # Default target resource
    CLASS_NAMES = ["tree"]  # in settings.py
    # Controls
    CLICK_DELAY = (0.2, 0.5)  # Random delay range
    
    # Debug
    DEBUG = True  # Enable debug overlays
settings = Settings()  # <-- Add this line
    