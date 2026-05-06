"""
YOLOv8 Human Detection Web App
Built with Streamlit
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from typing import Tuple, Optional


# Page configuration
st.set_page_config(
    page_title="YOLOv8 Human Detection",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(model_path: str, conf_thres: float = 0.5) -> Optional[YOLO]:
    """
    Load YOLOv8 model with caching

    Args:
        model_path: Path to model weights file
        conf_thres: Confidence threshold

    Returns:
        YOLO model or None if loading fails
    """
    try:
        if not Path(model_path).exists():
            st.error(f"Model file not found: {model_path}")
            return None

        model = YOLO(model_path)
        model.conf = conf_thres
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
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


def process_image(model: YOLO, image: np.ndarray,
                 confidence_threshold: float) -> Tuple[np.ndarray, int]:
    """
    Process image with YOLOv8 model

    Args:
        model: YOLOv8 model
        image: Input image
        confidence_threshold: Confidence threshold

    Returns:
        Tuple of (processed image, detection count)
    """
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

    return processed_image, detection_count


def main():
    """Main application"""

    # Header
    st.title("👤 YOLOv8 Human Detection")
    st.markdown("Detect humans in images using YOLOv8")

    # Sidebar
    st.sidebar.header("Settings")

    # Model settings
    model_path = st.sidebar.text_input(
        "Model Path",
        value="models/best.pt",
        help="Path to trained model weights"
    )

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence to display detections"
    )

    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_path, confidence_threshold)

    if model is None:
        st.error("Failed to load model. Please check the model path.")
        st.info("Make sure you have trained the model and the weights file exists.")
        return

    st.success("Model loaded successfully!")

    # Main content
    st.header("Upload Image")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "mp4"],
        help="Supported formats: JPG, JPEG, PNG, MP4"
    )

    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
        with col3:
            st.metric("File Type", uploaded_file.type)

        # Check file type
        if uploaded_file.type.startswith("image"):
            st.subheader("Image Detection")

            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Image**")
                st.image(image, use_column_width=True)

            # Process image
            with st.spinner("Processing image..."):
                processed_image, detection_count = process_image(
                    model, image_np, confidence_threshold
                )

            # Display results
            with col2:
                st.write("**Detection Result**")
                st.image(processed_image, use_column_width=True)

            # Display statistics
            st.subheader("Detection Statistics")
            st.metric("People Detected", detection_count)

            # Download button
            processed_pil = Image.fromarray(processed_image)
            st.download_button(
                label="Download Detection Result",
                data=processed_pil,
                file_name=f"detected_{uploaded_file.name}",
                mime="image/jpeg"
            )

        elif uploaded_file.type.startswith("video"):
            st.subheader("Video Detection")
            st.info("Video processing is coming soon! For now, please upload images.")

    # Instructions
    with st.expander("How to use"):
        st.markdown("""
        ### Instructions:
        1. **Upload an image**: Click the "Browse files" button above
        2. **Adjust settings**: Use the sidebar to adjust confidence threshold
        3. **View results**: See the detection results and statistics
        4. **Download**: Download the processed image with detections

        ### Tips:
        - Lower confidence threshold = more detections (may include false positives)
        - Higher confidence threshold = fewer detections (more accurate)
        - The model works best with clear, well-lit images
        """)


if __name__ == "__main__":
    main()