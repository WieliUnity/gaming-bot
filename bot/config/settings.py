class Settings:
    #showing object detection overlay
    DEBUG: bool = False
    DEBUG_DIR: str = "debug_frames"  # Add this line
    
    
    # Screen capture
    MONITOR_REGION = {
        "top": 0,
        "left": 0,
        "width": 2560,
        "height": 1600
    }
    
    # Object detection
    CONFIDENCE_THRESHOLD = 0.5
    MODEL_PATH = "bot/models/best_w_trunks.onnx"
    CLASS_NAMES = ["tree", "trunk"]  # Uncommented and corrected
    PRIORITY_TARGETS = ["trunk","tree"]
    #CLASS_NAMES = ["tree"]
    #PRIORITY_TARGETS = ["tree"]
    #TARGET_CLASS = ["tree", "trunk"]
    # Controls
    CLICK_DELAY = (0.2, 0.5)  # Random delay range
    
    # Debug

settings = Settings()  # <-- Add this line
    