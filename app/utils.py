"""
Utility functions for YOLOv8 Human Detection App
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file path

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def resize_image(image: np.ndarray, max_size: int = 800) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio

    Args:
        image: Input image
        max_size: Maximum dimension size

    Returns:
        Resized image
    """
    height, width = image.shape[:2]

    if max(height, width) <= max_size:
        return image

    if height > width:
        new_height = max_size
        new_width = int(width * (max_size / height))
    else:
        new_width = max_size
        new_height = int(height * (max_size / width))

    return cv2.resize(image, (new_width, new_height))


def draw_detections(image: np.ndarray, boxes: List,
                   confidence_threshold: float = 0.5,
                   color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw detection boxes and labels on image

    Args:
        image: Input image
        boxes: List of detection boxes
        confidence_threshold: Minimum confidence to display
        color: Box color in RGB format

    Returns:
        Image with drawn detections
    """
    image_copy = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        confidence = box[4] if len(box) > 4 else 0.0

        if confidence >= confidence_threshold:
            # Draw box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"Human: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.rectangle(image_copy,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)

            cv2.putText(image_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image_copy


def count_detections(boxes: List, confidence_threshold: float = 0.5) -> int:
    """
    Count number of detections above confidence threshold

    Args:
        boxes: List of detection boxes
        confidence_threshold: Minimum confidence to count

    Returns:
        Number of detections
    """
    count = 0
    for box in boxes:
        confidence = box[4] if len(box) > 4 else 0.0
        if confidence >= confidence_threshold:
            count += 1
    return count


def convert_to_pil(image: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image

    Args:
        image: Input image as numpy array

    Returns:
        PIL Image
    """
    return Image.fromarray(image)


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save image to file

    Args:
        image: Image to save
        output_path: Output file path

    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename

    Args:
        filename: Input filename

    Returns:
        File extension (lowercase, without dot)
    """
    return Path(filename).suffix[1:].lower()


def is_image_file(filename: str, image_types: List[str]) -> bool:
    """
    Check if file is an image

    Args:
        filename: Input filename
        image_types: List of supported image extensions

    Returns:
        True if file is an image
    """
    ext = get_file_extension(filename)
    return ext in image_types


def is_video_file(filename: str, video_types: List[str]) -> bool:
    """
    Check if file is a video

    Args:
        filename: Input filename
        video_types: List of supported video extensions

    Returns:
        True if file is a video
    """
    ext = get_file_extension(filename)
    return ext in video_types