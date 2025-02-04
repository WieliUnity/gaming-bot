class Settings:
    #showing object detection overlay
    DEBUG: bool = True
    DEBUG_DIR: str = "debug_frames"  # Add this line
    
    
    # Screen capture
    MONITOR_REGION = {
        "top": 0,
        "left": 0,
        "width": 2560,
        "height": 1600
    }
    
    # Object detection
    CONFIDENCE_THRESHOLD = 0.2
    MODEL_PATH = "bot/models/trunk_nano.onnx"
    CLASS_NAMES = ["trunk"]  # Uncommented and corrected
    PRIORITY_TARGETS = ["trunk"]
    INPUT_SIZE = 640    # Model input size
    IOU_THRESHOLD = 0.3    # Non-max suppression threshold
    OVERLAY_ALPHA = 0.7 # Bounding box transparency    

    
    # Target selection
    ROTATION_THRESHOLD = 30       # Pixels from center to trigger rotation
    MIN_TARGET_WIDTH = 40         # Minimum width (pixels) to initiate interaction
    MAX_TRACKING_TIME = 15        # Seconds before abandoning a target

    ICON_TEMPLATE_PATH = r"C:\Python Projects\gaming-bot\Icons\interaction.png"
    ICON_SEARCH_REGION = (1100, 1170, 100, 130)  # (x, y, w, h)
    ICON_COLOR = (26, 26, 26)     # RGB color to detect for interaction prompt

    ZONE_BOUNDARIES = {
        'left': 0.33,
        'center': 0.66 
    }

    # Target Selection Weights (confidence, size, centrality)
    TARGET_SCORE_WEIGHTS = (0.5, 0.2, 0.3) 
    
    # Interaction Timing
    MAX_WALK_TIME = 10.0          # From _perform_interaction()
    POST_INTERACTION_DELAY = 15   # Seconds after pressing 'F'
    
    
    # Controls
    CLICK_DELAY = (0.2, 0.5)  # Random delay range




    # Main loop
    MAX_DETECTION_WORKERS = 4
    
    # Debug

settings = Settings()  # <-- Add this line
    