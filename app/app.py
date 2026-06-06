"""Smart Store People Counting System built with Streamlit and YOLOv8."""

import csv
import io
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


st.set_page_config(
    page_title="Smart Store People Counting System",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model(model_path: str) -> Optional[YOLO]:
    """Load YOLOv8 model with Streamlit caching."""
    try:
        if not Path(model_path).exists():
            st.error(f"Model file not found: {model_path}")
            return None
        return YOLO(model_path)
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        return None


def draw_detections(
    image: np.ndarray,
    boxes: list,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Draw detection boxes and labels on image."""
    image_copy = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        confidence = float(box[4]) if len(box) > 4 else 0.0

        if confidence < confidence_threshold:
            continue

        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Human: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            image_copy,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            image_copy,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    return image_copy


def process_image(
    model: YOLO,
    image: np.ndarray,
    confidence_threshold: float,
    iou_threshold: float,
) -> Tuple[np.ndarray, int]:
    """Process an image with YOLOv8 model."""
    results = model(image, conf=confidence_threshold, iou=iou_threshold, verbose=False)

    boxes = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            boxes.append([x1, y1, x2, y2, confidence])

    processed_image = draw_detections(image, boxes, confidence_threshold)
    detection_count = len([b for b in boxes if b[4] >= confidence_threshold])
    return processed_image, detection_count


def save_uploaded_file(uploaded_file, output_dir: Path) -> Path:
    """Save a Streamlit uploaded file and return the saved path."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Cannot create output directory: {e}")
        raise
    
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    safe_stem = Path(uploaded_file.name).stem.replace(" ", "_")
    file_path = output_dir / f"{safe_stem}_{int(time.time())}{suffix}"

    try:
        with file_path.open("wb") as f:
            f.write(uploaded_file.getbuffer())
    except PermissionError as e:
        st.error(f"Permission denied saving file. Try using a different output directory or check disk permissions: {e}")
        raise
    except Exception as e:
        st.error(f"Error saving file: {e}")
        raise

    return file_path


def event_log_to_csv_bytes(event_log: list) -> bytes:
    """Convert event log list of dicts to CSV bytes."""
    output = io.StringIO()
    fieldnames = ["track_id", "direction", "frame_index", "timestamp_seconds"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(event_log)
    return output.getvalue().encode("utf-8")


def render_image_mode(uploaded_file, model_path: str, confidence_threshold: float, iou_threshold: float) -> None:
    """Render image detection workflow."""
    st.subheader("Image Detection")
    model = load_model(model_path)
    if model is None:
        st.stop()

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Processing image..."):
        processed_image, detection_count = process_image(
            model,
            image_np,
            confidence_threshold,
            iou_threshold,
        )

    with col2:
        st.write("Detection Result")
        st.image(processed_image, use_container_width=True)

    st.metric("People Detected", detection_count)

    processed_pil = Image.fromarray(processed_image)
    buf = io.BytesIO()
    processed_pil.save(buf, format="JPEG")
    buf.seek(0)
    st.download_button(
        label="Download Detection Result",
        data=buf,
        file_name=f"detected_{Path(uploaded_file.name).stem}.jpg",
        mime="image/jpeg",
    )


def render_video_mode(uploaded_file, model_path: str, confidence_threshold: float, iou_threshold: float) -> None:
    """Render video people-counting workflow."""
    from scripts.process_video import get_video_metadata, process_video
    
    st.subheader("Video People Counting")

    if not Path(model_path).exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    upload_path = None
    
    # Try to save to project outputs directory first
    try:
        upload_path = save_uploaded_file(uploaded_file, PROJECT_ROOT / "outputs" / "uploads")
    except Exception:
        # Fallback to temp directory if project outputs has permission issues
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "yolov8_uploads"
        st.warning(f"Could not save to project directory. Using temp directory: {temp_dir}")
        upload_path = save_uploaded_file(uploaded_file, temp_dir)

    try:
        metadata = get_video_metadata(str(upload_path))
    except Exception as exc:
        st.error(f"Cannot read video: {exc}")
        st.stop()

    video_width = int(metadata["width"])
    video_height = int(metadata["height"])
    total_frames = int(metadata["frame_count"])
    duration = float(metadata["duration_seconds"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Resolution", f"{video_width} x {video_height}")
    col2.metric("Frames", total_frames)
    col3.metric("Duration", f"{duration:.1f}s")

    st.video(str(upload_path))

    st.sidebar.divider()
    st.sidebar.subheader("Counting Settings")

    tracker = st.sidebar.selectbox(
        "Tracker",
        ["bytetrack.yaml", "botsort.yaml"],
        index=0,
    )

    line_orientation = st.sidebar.selectbox(
        "Line orientation",
        ["horizontal", "vertical"],
        index=0,
    )

    if line_orientation == "horizontal":
        line_position = st.sidebar.slider(
            "Line Y",
            min_value=0,
            max_value=max(video_height, 1),
            value=video_height // 2,
        )
    else:
        line_position = st.sidebar.slider(
            "Line X",
            min_value=0,
            max_value=max(video_width, 1),
            value=video_width // 2,
        )

    crowd_threshold = st.sidebar.number_input(
        "Crowd threshold",
        min_value=1,
        value=10,
        step=1,
    )

    output_path = PROJECT_ROOT / "outputs" / "videos" / f"counted_{upload_path.stem}.mp4"

    if st.button("🚀 Process Video", type="primary"):
        progress_bar = st.progress(0)
        status = st.empty()

        def update_progress(progress: float) -> None:
            progress_bar.progress(progress)
            status.write(f"Processing video... {progress * 100:.1f}%")

        try:
            with st.spinner("Processing video with YOLOv8 tracking..."):
                result = process_video(
                    source_path=str(upload_path),
                    model_path=model_path,
                    output_path=str(output_path),
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    tracker=tracker,
                    line_orientation=line_orientation,
                    line_position=line_position,
                    progress_callback=update_progress,
                )
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
            st.stop()

        progress_bar.progress(1.0)
        status.success("Processing complete!")

        in_count = int(result["in_count"])
        out_count = int(result["out_count"])
        current_occupancy = int(result["current_occupancy"])
        event_log = result.get("event_log", [])

        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("People In", in_count)
        metric2.metric("People Out", out_count)
        metric3.metric("Current Occupancy", current_occupancy)

        if current_occupancy >= crowd_threshold:
            metric4.metric("Crowd Status", "Warning")
            st.warning("Crowd Alert: occupancy is above the configured threshold.")
        else:
            metric4.metric("Crowd Status", "Normal")
            st.success("Normal: occupancy is below the configured threshold.")

        st.subheader("Annotated Output Video")
        st.video(str(output_path))

        with output_path.open("rb") as f:
            st.download_button(
                label="Download Annotated Video",
                data=f,
                file_name=output_path.name,
                mime="video/mp4",
            )

        st.subheader("Crossing Event Log")
        if event_log:
            st.dataframe(event_log, use_container_width=True)
            st.download_button(
                label="Download CSV Event Log",
                data=event_log_to_csv_bytes(event_log),
                file_name=f"events_{upload_path.stem}.csv",
                mime="text/csv",
            )
        else:
            st.info("No crossing events were detected. Try adjusting the line position or use another video.")


def main() -> None:
    """Main Streamlit application."""
    st.title("🏪 Smart Store People Counting System")
    st.markdown("YOLOv8 + multi-object tracking + line crossing for store entrance analytics.")

    st.sidebar.header("Model Settings")
    model_path = st.sidebar.text_input(
        "Model Path",
        value="models/best.pt",
        help="Path to trained model weights",
    )

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
    )

    iou_threshold = st.sidebar.slider(
        "IOU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
    )

    uploaded_file = st.file_uploader(
        "Upload image or video",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    )

    if uploaded_file is None:
        st.info("Upload an image for human detection or a video for IN/OUT people counting.")
        with st.expander("How to use"):
            st.markdown(
                """
                1. Upload a store entrance video.
                2. Choose tracker and line orientation in the sidebar.
                3. Adjust the line position to match the entrance.
                4. Click **Process Video**.
                5. Download the annotated video and CSV event log.
                """
            )
        return

    info1, info2, info3 = st.columns(3)
    info1.metric("File Name", uploaded_file.name)
    info2.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
    info3.metric("File Type", uploaded_file.type or "unknown")

    if uploaded_file.type and uploaded_file.type.startswith("image"):
        render_image_mode(uploaded_file, model_path, confidence_threshold, iou_threshold)
    elif uploaded_file.type and uploaded_file.type.startswith("video"):
        render_video_mode(uploaded_file, model_path, confidence_threshold, iou_threshold)
    else:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix in [".jpg", ".jpeg", ".png"]:
            render_image_mode(uploaded_file, model_path, confidence_threshold, iou_threshold)
        elif suffix in [".mp4", ".avi", ".mov"]:
            render_video_mode(uploaded_file, model_path, confidence_threshold, iou_threshold)
        else:
            st.error("Unsupported file type.")


if __name__ == "__main__":
    main()
