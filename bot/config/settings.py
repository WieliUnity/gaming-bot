class Settings:
    #showing object detection overlay
    SHOW_OVERLAY: bool = True
    ROBOFLOW_API_KEY: str = "sCfrom9qV4tNDXpgFScX"  # Your key here
    
    # Screen capture
    MONITOR_REGION = {
        "top": 0,
        "left": 0,
        "width": 1920,
        "height": 1080
    }
    
    # Object detection
    CONFIDENCE_THRESHOLD = 0.7
    MODEL_PATH = "bot/models/tree_model.onnx"
    TARGET_CLASS = "tree"  # Default target resource
    
    # Controls
    CLICK_DELAY = (0.2, 0.5)  # Random delay range
    
    # Debug
    DEBUG = True  # Enable debug overlays
settings = Settings()  # <-- Add this line
    