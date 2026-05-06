"""
YOLOv8 Human Detection - CLI Prediction Tool
Run predictions on single images from command line
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Optional


def load_model(model_path: str, conf_thres: float = 0.5) -> Optional[YOLO]:
    """
    Load YOLOv8 model

    Args:
        model_path: Path to model weights file
        conf_thres: Confidence threshold

    Returns:
        YOLO model or None if loading fails
    """
    try:
        if not Path(model_path).exists():
            print(f"Error: Model file not found: {model_path}")
            return None

        model = YOLO(model_path)
        model.conf = conf_thres
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def draw_detections(image: np.ndarray, boxes: list,
                   confidence_threshold: float = 0.5) -> np.ndarray:
    """
    Draw detection boxes and labels on image

    Args:
        image: Input image
        boxes: List of detection boxes
        confidence_threshold: Minimum confidence to display

    Returns:
        Image with drawn detections
    """
    image_copy = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        confidence = box[4] if len(box) > 4 else 0.0

        if confidence >= confidence_threshold:
            # Draw box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"Human: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.rectangle(image_copy,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         (0, 255, 0), -1)

            cv2.putText(image_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image_copy


def predict_image(model: YOLO, image_path: str,
                 output_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 show_result: bool = False) -> int:
    """
    Run prediction on single image

    Args:
        model: YOLOv8 model
        image_path: Path to input image
        output_path: Path to save output image (optional)
        confidence_threshold: Confidence threshold
        show_result: Whether to display result window

    Returns:
        Number of detections
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return 0

        # Run inference
        results = model(image, conf=confidence_threshold, iou=0.45)

        # Get detections
        boxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    boxes.append([x1, y1, x2, y2, confidence])

        # Draw detections
        processed_image = draw_detections(image, boxes, confidence_threshold)

        # Count detections
        detection_count = len([b for b in boxes if b[4] >= confidence_threshold])

        print(f"Detected {detection_count} human(s) in {image_path}")

        # Save result
        if output_path:
            cv2.imwrite(output_path, processed_image)
            print(f"Result saved to: {output_path}")

        # Show result
        if show_result:
            cv2.imshow("Detection Result", processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detection_count

    except Exception as e:
        print(f"Error during prediction: {e}")
        return 0


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="YOLOv8 Human Detection - CLI Prediction Tool"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="C:\\Users\\fptsh\\Downloads\\Test app\\project\\yolov8-human-detector\\models\\best.pt",
        help="Path to model weights file (default: models/best.pt)"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output image (optional)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show result in window"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model}")
    model = load_model(args.model, args.conf)

    if model is None:
        print("Failed to load model. Exiting.")
        return

    print("Model loaded successfully!")

    # Run prediction
    detection_count = predict_image(
        model,
        args.source,
        args.output,
        args.conf,
        args.show
    )

    print(f"Total detections: {detection_count}")


if __name__ == "__main__":
    main()