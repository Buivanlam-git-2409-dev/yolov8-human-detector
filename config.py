"""
Configuration file for YOLOv8 Human Detection App
"""

class Config:
    """Application configuration"""

    # Model settings
    MODEL_PATH = "runs/detect/train/weights/best.pt"
    MODEL_SIZE = 640
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45

    # UI settings
    APP_TITLE = "YOLOv8 Human Detection"
    APP_DESCRIPTION = "Detect humans in images and videos using YOLOv8"

    # Supported file types
    IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
    VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]

    # Display settings
    MAX_IMAGE_SIZE = 800
    MAX_VIDEO_WIDTH = 800

    # Colors for visualization
    COLORS = {
        'human': (0, 255, 0),  # Green
        'box': (0, 255, 0),
        'text': (255, 255, 255)
    }